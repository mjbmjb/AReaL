import asyncio
import copy
import uuid
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

from .tool_manager import ToolCallStatus, ToolManager

logger = logging.getLogger("TIR workflow")

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to help the user.
You can use the following tools:
{tool_descriptions}
When you invoke a tool in your response, the tool's output will be immediately obtained and placed within the output``` ``` tags. Then, you continue answering based on the tool's output. Depending on the parameters you provide for the invocation, the tool's invocation may fail. You can invoke the tool multiple times in your response.
You should use the tools to help the user to solve the problem whenever possible. 
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

BASE_MODEL_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls multiple tools to find get necessary information, and provides the user with the answer. 
The reasoning processes are enclosed within <think> </think>.
The available tools are:
{tool_descriptions}

Finally, the Assistant provides answer within \\boxed{{}}., i.e. \\boxed{{4}}. 

User: 
{question}

Assistant:
<think>"""


ANSWER = r"\boxed{.*?}"

class TIRWorkflow(RolloutWorkflow):
    """Tool-Integrated Reasoning Workflow for multi-turn tool calling."""
    
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        tool_manager: ToolManager,
        chat_model: bool = False,
        max_turns: int = 2,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: Optional[str] = None,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.tool_manager = tool_manager
        self.chat_model = chat_model
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
        return f"\n```output\n{tool_result}\n```\n"
    
    async def arun_episode(self, engine: InferenceEngine, data: Dict[str, Any]) -> TensorDict:
        """Run a complete TIR inference episode.
        :param engine: The inference engine.
        :param data: The input data.
        :return: The output tensor dict.
        """
        # 初始化对话历史
        messages = data["messages"]

        # 添加system prompt，添加工具使用的prompt
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=self.tool_manager.get_tool_descriptions_prompt())
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", 
                                "content": system_prompt})
        
        # 准备输入
        if self.chat_model:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            input_ids = self.tokenizer.encode(BASE_MODEL_PROMPT.format(question=messages[1]["content"], 
                                                                       tool_descriptions=self.tool_manager.get_tool_descriptions_prompt()), add_special_tokens=False)

        n_samples = self.gconfig.n_samples
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
        stop_reason = None
        # 多轮推理循环
        max_len = 4096
        hit_max_len = False
        turn = 0
        # 每个episode的状态标记：是否在等待工具开始标记
        waiting_for_tool_start = True
        tool_start_idx = -1
        while turn <= self.max_turns:
            if len(context_ids) >= max_len-1:
                hit_max_len = True
                break

            # 生成响应
            resp, stop_reason = await self._generate_response(engine, context_ids, max_len, waiting_for_tool_start)

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
        
            # 结束token, 截断
            if context_ids[-1] in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                break

            # 如果出现答案, 立刻截断
            if re.search(ANSWER, cur_completions_str):
                break

            # 状态转换逻辑：检测是否遇到了工具开始标记
            if waiting_for_tool_start and stop_reason == "stop":
                # 检查是否检测到工具开始标记
                tool_start_marker = self._detect_tool_start_marker(cur_completions_str)
                if tool_start_marker:
                    logger.info(f"🔧 Detected tool start marker {tool_start_marker}, switching to end marker mode {cur_completions_str}")
                    waiting_for_tool_start = False
                    tool_start_idx = len(completions_str) - len(tool_start_marker)
                    # 继续生成到工具结束标记
                    continue

            # 如果检测到工具调用，执行工具调用
            if not waiting_for_tool_start and stop_reason == "stop" and tool_start_idx != -1:
                logger.info(f"hit stop token {completions_str[tool_start_idx:]}")
                tool_results, tool_status = self._execute_tools(completions_str[tool_start_idx:])
                if tool_status == ToolCallStatus.NOT_FOUND:
                    # 没匹配上, 继续生成到下一个工具结束标记
                    continue
                turn += 1
                has_tool = True
                tool_call_count += 1  # 增加工具调用计数
                tool_success_count += 1 if tool_status else 0
                tool_results = self._process_tool_result(tool_results)
                # append tool_response_ids
                tool_rsp_token_ids=self.tokenizer.encode(tool_results, add_special_tokens=False)
                # 拼接到seq上
                # 构建tool mask
                context_ids.extend(tool_rsp_token_ids)
                seq.extend(tool_rsp_token_ids)
                logprobs.extend([0.0] * len(tool_rsp_token_ids))
                loss_mask.extend([0] * len(tool_rsp_token_ids))
                versions.extend([-1] * len(tool_rsp_token_ids))
                completions_str += tool_results
                
                # 工具执行完成后，重置状态标记，准备检测下一个工具调用
                waiting_for_tool_start = True
                logger.info(f"🔄 Tool execution completed, reset to start marker mode tool result {tool_results}, completions_str {completions_str}")
        
        
        # 为base模型添加eos token
        # if stop_reason != 'length' and seq[-1] not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
        #     seq.append(self.tokenizer.eos_token_id)
        #     logprobs.append(0.0)
        #     loss_mask.append(1)
        #     versions.append(-1)

        if has_tool:
            logger.info(f"all seq {completions_str}")

        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data
        )
        logger.info(f"💰 Final reward: {reward} stop reason {stop_reason} hit_max_len {hit_max_len} {len(context_ids)} with {completions_str}")
        
        # 记录工具调用次数到stats_tracker
        stats_tracker.get(self.rollout_stat_scope).scalar(
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count
        )

        res = dict(
            input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
            logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
            versions=torch.tensor(versions[:max_len]).unsqueeze(0),
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(0),
            rewards=torch.tensor([float(reward)]),
        )
        return TensorDict(res, batch_size=[1])

    async def _generate_response(self, engine: InferenceEngine, input_ids: list[int], max_len: int, waiting_for_tool_start: bool) -> Tuple[ModelResponse, str]:
        """生成响应，支持工具调用检测"""
        
        # 根据状态标记选择停止条件
        if waiting_for_tool_start:
            # 等待工具开始标记时，使用start_markers停止
            stop_markers = [marker for marker in self.start_markers]
            # logger.info("🔍 Waiting for tool start markers")
        else:
            # 已检测到工具开始，使用end_markers停止
            stop_markers = [marker for marker in self.end_markers]
            # logger.info("🔍 Waiting for tool end markers")
        
        # 设置生成配置，添加工具调用停止token
        gconfig = self.gconfig.new(
            n_samples=1,
            stop=[marker for marker in stop_markers],
            max_new_tokens=min(self.gconfig.max_new_tokens, max_len - len(input_ids) - 1), # 相同的话会报错, 需要-1
        )
        
        # 生成响应
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        return resp, resp.stop_reason
    
    def post_process_stop_reason(self, text: str, stop_reason: str) -> bool:
        """检测是由于工具调用结束"""
        if stop_reason == "stop":
            # 检测是否有工具调用结束标记
            if any(text.endswith(marker) for marker in self.end_markers):
                logger.info(f"🔍 Detected tool call: {text[-10:]}")
                return "tool_call"
        return stop_reason
    
    def _detect_tool_start_marker(self, text: str) -> Optional[str]:
        """检测文本末尾是否包含工具开始标记"""
        for marker in self.start_markers:
            if text.endswith(marker):
                logger.info(f"🎯 Found tool start marker at end: {marker}")
                return marker
        return None
    
    def _execute_tools(self, response: str) -> str:
        """执行工具调用"""
        # 调用execute_tool_call
        tool_results = self.tool_manager.execute_tool_call(response)
        return tool_results
