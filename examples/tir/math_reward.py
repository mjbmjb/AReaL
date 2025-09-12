import re
from typing import Any, Dict, List, Optional
import math

from areal.utils import logging

logger = logging.getLogger("Math Reward")


class MathRewardFunction:
    """数学推理奖励函数"""
    
    def __init__(self):
        self.answer_patterns = [
            r"the answer is\s*([^\n]+)",
            r"final answer:\s*([^\n]+)",
            r"result:\s*([^\n]+)",
            r"solution:\s*([^\n]+)",
            r"answer:\s*([^\n]+)",
            r"=\s*([^\n]+)",
        ]
    
    def __call__(self, prompt: str, completions: str, **kwargs) -> float:
        """计算奖励分数"""
        logger.info("🎯 Starting reward calculation")
        logger.debug(f"📝 Completions: {completions[:200]}...")
        
        try:
            # 提取预测答案
            logger.debug("🔍 Extracting predicted answer")
            predicted_answer = self._extract_answer(completions)
            logger.info(f"🎯 Predicted answer: '{predicted_answer}'")
            
            if not predicted_answer:
                logger.warning("⚠️ No predicted answer found")
                return 0.0
            
            # 获取真实答案
            ground_truth = kwargs.get("answer", "")
            logger.info(f"✅ Ground truth: '{ground_truth}'")
            
            if not ground_truth:
                logger.warning("⚠️ No ground truth provided")
                return 0.0
            
            # 计算奖励
            logger.debug("🧮 Calculating reward")
            reward = self._calculate_reward(predicted_answer, ground_truth)
            logger.info(f"💰 Final reward: {reward}")
            return reward
            
        except Exception as e:
            logger.error(f"❌ Error calculating reward: {e}")
            return 0.0
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """从文本中提取答案"""
        text = text.strip()
        
        # 尝试各种模式匹配
        for pattern in self.answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # 清理答案
                answer = self._clean_answer(answer)
                if answer:
                    return answer
        
        # 如果没有找到明确的答案标记，尝试提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _clean_answer(self, answer: str) -> str:
        """清理答案文本"""
        # 移除常见的后缀
        suffixes_to_remove = [
            ".", ",", ";", "!", "?", ":", ")", "]", "}",
            " dollars", " $", " units", " items", " people",
            " years", " days", " hours", " minutes", " seconds"
        ]
        
        for suffix in suffixes_to_remove:
            if answer.endswith(suffix):
                answer = answer[:-len(suffix)].strip()
        
        return answer.strip()
    
    def _calculate_reward(self, predicted: str, ground_truth: str) -> float:
        """计算奖励分数"""
        logger.debug(f"🧮 Comparing '{predicted}' vs '{ground_truth}'")
        
        try:
            # 尝试数值比较
            if self._is_numeric(predicted) and self._is_numeric(ground_truth):
                pred_num = float(predicted)
                truth_num = float(ground_truth)
                logger.debug(f"🔢 Numeric comparison: {pred_num} vs {truth_num}")
                
                # 检查是否相等（考虑浮点数精度）
                if abs(pred_num - truth_num) < 1e-6:
                    logger.info("✅ Numeric match found")
                    return 1.0
                else:
                    logger.info("❌ Numeric values don't match")
                    return 0.0
            
            # 尝试字符串比较
            if predicted.lower().strip() == ground_truth.lower().strip():
                logger.info("✅ Exact string match found")
                return 1.0
            
            # 尝试部分匹配
            if self._is_partial_match(predicted, ground_truth):
                logger.info("🔍 Partial match found")
                return 0.5
            
            logger.info("❌ No match found")
            return 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error in reward calculation: {e}")
            return 0.0
    
    def _is_numeric(self, text: str) -> bool:
        """检查文本是否为数字"""
        try:
            float(text)
            return True
        except ValueError:
            return False
    
    def _is_partial_match(self, predicted: str, ground_truth: str) -> bool:
        """检查部分匹配"""
        pred_clean = re.sub(r'[^\w\s]', '', predicted.lower())
        truth_clean = re.sub(r'[^\w\s]', '', ground_truth.lower())
        
        # 检查是否包含关键数字
        pred_numbers = re.findall(r'\d+\.?\d*', predicted)
        truth_numbers = re.findall(r'\d+\.?\d*', ground_truth)
        
        if pred_numbers and truth_numbers:
            return any(p in truth_numbers for p in pred_numbers)
        
        return False


class MathAnswerParser:
    """数学答案解析器"""
    
    def __init__(self):
        self.patterns = [
            r"(\d+\.?\d*)\s*$",  # 以数字结尾
            r"(\d+\.?\d*)\s*[^\d]",  # 数字后跟非数字字符
            r"(\d+\.?\d*)\s*[a-zA-Z]",  # 数字后跟字母
        ]
    
    def parse(self, text: str) -> Optional[float]:
        """解析文本中的数字答案"""
        for pattern in self.patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None


class SympyParser:
    """SymPy符号计算解析器（简化版）"""
    
    def __init__(self):
        self.supported_operations = ['+', '-', '*', '/', '**', '^']
    
    def parse(self, expression: str) -> Optional[str]:
        """解析数学表达式"""
        try:
            # 简单的表达式清理
            expr = expression.strip()
            
            # 替换常见的数学符号
            expr = expr.replace('^', '**')
            expr = expr.replace('×', '*')
            expr = expr.replace('÷', '/')
            
            # 检查是否包含支持的运算
            if any(op in expr for op in self.supported_operations):
                return expr
            
            return None
            
        except Exception:
            return None
