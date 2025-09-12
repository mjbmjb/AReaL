import asyncio
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

from .tool_manager import ToolManager
from .math_reward import MathRewardFunction

logger = logging.getLogger("TIR workflow")


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
        
        # 工具调用相关的特殊token
        self.tool_tokens = self._setup_tool_tokens()
        
    def _setup_tool_tokens(self) -> Dict[str, int]:
        """设置工具调用相关的特殊token ID"""
        # 这些token需要在tokenizer中存在，或者我们需要添加它们
        tool_tokens = {
            "tool_call_start": "<tool_call>",
            "tool_call_end": "</tool_call>",
            "python_start": "<python>",
            "python_end": "</python>",
            "tool_result_start": "<tool_result>",
            "tool_result_end": "</tool_result>",
        }
        
        # 获取token ID，如果不存在则使用特殊token
        token_ids = {}
        for key, token in tool_tokens.items():
            if token in self.tokenizer.get_vocab():
                token_ids[key] = self.tokenizer.convert_tokens_to_ids(token)
            else:
                # 使用EOS token作为fallback
                token_ids[key] = self.tokenizer.eos_token_id
                
        return token_ids
    
    async def arun_episode(self, engine: InferenceEngine, data: Dict[str, Any]) -> TensorDict:
        """运行一个完整的TIR推理episode"""
        # 初始化对话历史
        messages = data["messages"]
        conversation_history = []
        
        # 多轮推理循环
        for turn in range(self.max_turns):
            logger.info(f"TIR Turn {turn + 1}/{self.max_turns}")
            
            # 生成响应
            response = await self._generate_response(engine, messages)
            conversation_history.append({
                "turn": turn,
                "response": response,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 检查是否包含工具调用
            if self._has_tool_call(response):
                logger.info(f"Tool call detected in turn {turn + 1}")
                # 执行工具调用
                tool_results = await self._execute_tools(response)
                
                # 将工具结果整合到对话中
                messages = self._integrate_tool_results(messages, tool_results)
                conversation_history[-1]["tool_results"] = tool_results
            else:
                # 没有工具调用，检查是否是最终答案
                if self._is_final_answer(response):
                    logger.info(f"Final answer reached in turn {turn + 1}")
                    break
        
        # 计算奖励
        reward = await self._calculate_reward(conversation_history, data)
        
        # 格式化轨迹数据
        return self._format_trajectory(conversation_history, reward)
    
    async def _generate_response(self, engine: InferenceEngine, messages: List[Dict]) -> str:
        """生成响应，支持工具调用检测"""
        # 准备输入
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        # 设置生成配置，添加工具调用停止token
        gconfig = self.gconfig.new(
            stop_token_ids=[self.tool_tokens["tool_call_start"]],
            max_new_tokens=min(self.gconfig.max_new_tokens, 512)  # 限制单次生成长度
        )
        
        # 生成响应
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        response_text = self.tokenizer.decode(resp.output_tokens)
        
        # 如果检测到工具调用，继续生成工具调用内容
        if self._has_tool_call(response_text):
            tool_content = await self._generate_tool_call(engine, resp)
            response_text += tool_content
            
        return response_text
    
    async def _generate_tool_call(self, engine: InferenceEngine, prev_resp) -> str:
        """生成工具调用内容"""
        # 继续从工具调用token开始生成
        gconfig = self.gconfig.new(
            max_new_tokens=256,  # 工具调用内容通常较短
            stop_token_ids=[self.tool_tokens["tool_call_end"]]
        )
        
        # 构建包含工具调用开始token的输入
        tool_start_token = self.tokenizer.convert_tokens_to_ids(self.tool_tokens["tool_call_start"])
        input_ids = prev_resp.input_tokens + prev_resp.output_tokens + [tool_start_token]
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        return self.tokenizer.decode(resp.output_tokens)
    
    def _has_tool_call(self, text: str) -> bool:
        """检测文本中是否包含工具调用"""
        tool_patterns = [
            r"<tool_call>",
            r"<python>",
            r"```python",
            r"def\s+\w+\s*\(",
            r"import\s+\w+",
            r"print\s*\(",
        ]
        
        for pattern in tool_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    async def _execute_tools(self, response: str) -> List[Dict[str, Any]]:
        """执行工具调用"""
        tool_results = []
        
        # 提取Python代码
        python_code = self._extract_python_code(response)
        if python_code:
            result = await self.tool_manager.execute_python(python_code)
            tool_results.append({
                "tool": "python",
                "code": python_code,
                "result": result,
                "success": not result.startswith("Error:")
            })
        
        return tool_results
    
    def _extract_python_code(self, text: str) -> Optional[str]:
        """从文本中提取Python代码"""
        # 匹配多种Python代码格式
        patterns = [
            r"<python>(.*?)</python>",
            r"```python\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _integrate_tool_results(self, messages: List[Dict], tool_results: List[Dict]) -> List[Dict]:
        """将工具结果整合到对话中"""
        if not tool_results:
            return messages
        
        # 构建工具结果文本
        tool_result_text = "\n\nTool Results:\n"
        for result in tool_results:
            tool_result_text += f"- {result['tool']}: {result['result']}\n"
        
        # 添加到对话历史
        messages.append({
            "role": "assistant",
            "content": tool_result_text
        })
        
        return messages
    
    def _is_final_answer(self, response: str) -> bool:
        """判断是否是最终答案"""
        # 简单的启发式规则
        final_indicators = [
            "the answer is",
            "final answer:",
            "result:",
            "solution:",
            "therefore",
            "thus",
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in final_indicators)
    
    async def _calculate_reward(self, conversation_history: List[Dict], data: Dict[str, Any]) -> float:
        """计算奖励"""
        # 提取最终响应
        final_response = conversation_history[-1]["response"] if conversation_history else ""
        
        # 使用奖励函数计算奖励
        reward = await self.async_reward_fn(
            prompt="",  # 这里可以传入完整的prompt
            completions=final_response,
            prompt_ids=[],
            completion_ids=[],
            **data
        )
        
        # 记录奖励
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        
        return reward
    
    def _format_trajectory(self, conversation_history: List[Dict], reward: float) -> TensorDict:
        """格式化轨迹数据为AReaL格式"""
        # 构建完整的序列
        full_sequence = []
        logprobs = []
        loss_mask = []
        versions = []
        
        # 这里简化处理，实际应该从conversation_history中提取token信息
        # 为了简化，我们使用一个基本的序列
        for turn_data in conversation_history:
            response = turn_data["response"]
            # 这里应该将response转换为token序列
            # 为了简化，我们使用一个占位符
            full_sequence.extend([1, 2, 3])  # 占位符token
            logprobs.extend([0.0, 0.0, 0.0])  # 占位符logprob
            loss_mask.extend([1, 1, 1])  # 占位符loss mask
            versions.extend([-1, -1, -1])  # 占位符version
        
        # 构建结果
        result = dict(
            input_ids=torch.tensor(full_sequence).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(full_sequence), dtype=torch.bool).unsqueeze(0),
            rewards=torch.tensor([float(reward)]),
        )
        
        return TensorDict(result, batch_size=[1])
