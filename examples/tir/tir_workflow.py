import asyncio
import copy
import re
import uuid
from typing import Dict, List, Optional, Any

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

from .tool_manager import ToolManager, ToolCallStatus
from .math_reward import MathRewardFunction

logger = logging.getLogger("TIR workflow")

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to help the user.
You can use the following tools:
{tool_descriptions}
When you invoke a tool in your response, the tool's output will be immediately obtained and placed within the tool_result``` ``` tags. Then, you continue answering based on the tool's output. Depending on the parameters you provide for the invocation, the tool's invocation may fail. You can invoke the tool multiple times in your response.
You should use the tools to help the user to solve the problem whenever possible. 
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

class TIRWorkflow(RolloutWorkflow):
    """Tool-Integrated Reasoning Workflow for multi-turn tool calling."""
    
    def __init__(
        self,
        reward_fn: MathRewardFunction,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        tool_manager: ToolManager,
        max_turns: int = 5,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: Optional[str] = None,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.tool_manager = tool_manager
        self.max_turns = max_turns
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        
        self.start_markers = self.tool_manager.get_all_start_markers()
        self.end_markers = self.tool_manager.get_all_end_markers()

        logger.info(f"start markers: {self.start_markers}, end markers {self.end_markers}")
    
    @staticmethod
    def _process_tool_result(tool_result: str) -> str:
        return f"\n```tool_result\n{tool_result}\n```\n"
    
    async def arun_episode(self, engine: InferenceEngine, data: Dict[str, Any]) -> TensorDict:
        """运行一个完整的TIR推理episode"""
        logger.info("🚀 Starting TIR episode")
        # logger.info(f"📝 Input data: {data.get('messages', [{}])[0].get('content', '')[:100]}...")
        # logger.info(f"🎯 Expected answer: {data.get('answer', 'N/A')}")
        
        # 初始化对话历史
        messages = data["messages"]

        # 添加system prompt，添加工具使用的prompt
        if messages[0]["role"] == "user":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT.format(tool_descriptions=self.tool_manager.get_tool_descriptions_prompt())})
        
        logger.info("🔧 Preparing input for generation")
        # 准备输入
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        logger.info(f"📏 Input token length: {len(input_ids)}")

        n_samples = self.gconfig.n_samples
        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []
        # append conversation_history
        results = await asyncio.gather(*[self._multi_round_response(engine, input_ids, data) for _ in range(n_samples)])
        
        return concat_padded_tensors(results)

    async def _multi_round_response(self, engine, prompt_ids, data):
        seq = []
        output_ids = []
        logprobs = []
        loss_mask = []
        versions = []
        prompt_str = self.tokenizer.decode(prompt_ids)
        context_ids = copy.deepcopy(prompt_ids)
        completions_str = ""
        has_tool = False
        tool_call_count = 0
        tool_success_count = 0
        # 多轮推理循环
        for turn in range(self.max_turns):
            logger.info(f"🔄 TIR Turn {turn + 1}/{self.max_turns}")            
            # 生成响应
            resp, stop_reason = await self._generate_response(engine, context_ids)
            logger.info(f"stop reason {stop_reason}")

            if turn == 0:
                # 第一轮, 后续轮次需要拼接到seq上
                seq = resp.input_tokens + resp.output_tokens
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
            else:
                context_ids.extend(resp.output_tokens)
                seq.extend(resp.output_tokens)
                logprobs.extend(resp.output_logprobs)
                loss_mask.extend([1] * resp.output_len)
                versions.extend(resp.output_versions)
            
            cur_completions_str = self.tokenizer.decode(resp.output_tokens)
            completions_str += cur_completions_str
            output_ids.extend(resp.output_tokens)
        
            logger.info(f"📤 Generated response: ..{completions_str[-100:]}")
            
            # 如果检测到工具调用，执行工具调用
            if stop_reason == "tool_call":
                tool_results, tool_status = self._execute_tools(cur_completions_str)
                if tool_status == ToolCallStatus.NOT_FOUND:
                    continue
                has_tool = True
                tool_call_count += 1  # 增加工具调用计数
                tool_success_count += 1 if tool_status else 0
                tool_results = self._process_tool_result(tool_results)
                # append tool_response_ids
                encoding=self.tokenizer(tool_results, add_special_tokens=False, return_offsets_mapping=True)
                tool_rsp_token_ids=encoding['input_ids']
                # 拼接到seq上
                # 构建tool mask
                context_ids.extend(tool_rsp_token_ids)
                seq.extend(tool_rsp_token_ids)
                logprobs.extend([0.0] * len(tool_rsp_token_ids))
                loss_mask.extend([0] * len(tool_rsp_token_ids))
                versions.extend([-1] * len(tool_rsp_token_ids))
                completions_str += tool_results
            else:
                # 生成结束
                break
        
        if has_tool:
            logger.info(f"all seq {self.tokenizer.decode(seq)}")

        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data
        )
        logger.info(f"💰 Final reward: {reward} with {completions_str}")
        
        # 记录工具调用次数到stats_tracker
        stats_tracker.get(self.rollout_stat_scope).scalar(
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count
        )
        logger.info(f"🔧 Tool calls made: {tool_call_count}")

        res = dict(
            input_ids=torch.tensor(seq).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            rewards=torch.tensor([float(reward)]),
        )
        return TensorDict(res, batch_size=[1])

    async def _generate_response(self, engine: InferenceEngine, input_ids: List[int]) -> str:
        """生成响应，支持工具调用检测"""
        
        # 设置生成配置，添加工具调用停止token
        gconfig = self.gconfig.new(
            n_samples=1,
            stop=[marker for marker in self.end_markers],
            max_new_tokens=self.gconfig.max_new_tokens
        )
        logger.debug(f"⚙️ Generation config: max_tokens={gconfig.max_new_tokens}, stop_tokens={gconfig.stop_token_ids}")
        
        # 生成响应
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        response_text = self.tokenizer.decode(resp.output_tokens)
        stop_reason = self.post_process_stop_reason(response_text, resp.stop_reason)

        return resp, stop_reason
    
    def post_process_stop_reason(self, text: str, stop_reason: str) -> bool:
        """检测是由于工具调用结束"""
        if stop_reason == "stop":
            # 检测是否有工具调用结束标记
            if any(text.endswith(marker) for marker in self.end_markers):
                logger.info(f"🔍 Detected tool call: {text[-10:]}")
                return "tool_call"
        return stop_reason
    
    def _execute_tools(self, response: str) -> str:
        """执行工具调用"""
        logger.info("🛠️ Starting tool execution")
        # 调用execute_tool_call
        tool_results = self.tool_manager.execute_tool_call(response)
        return tool_results
