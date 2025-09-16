import re
from typing import Dict, Any, Tuple

from areal.utils import logging
from .base import BaseTool, ToolType, ToolDescription, ToolCallStatus

logger = logging.getLogger("Calculator Tool")


class CalculatorTool(BaseTool):
    """基础计算器工具"""
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.CALCULATOR
    
    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python calculator",
            description="Perform basic mathematical calculations, supporting addition, subtraction, multiplication, division, and parentheses.",
            parameters={
                "expression": "Mathematical expression string"
            },
            parameter_prompt="Please provide a mathematical expression. Supports addition, subtraction, multiplication, division, and parentheses.",
            example="<calculator>1 + 2 * 3</calculator>"
        )
    
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """从<calculator>标记中提取数学表达式"""
        pattern = r"<calculator>(.*?)</calculator>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            expression = match.group(1).strip()
            logger.info(f"🧮 Extracted expression: {expression}")
            return {"expression": expression}
        else:
            logger.warning("⚠️ No <calculator> tag found")
            return {"expression": ""}
    
    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """执行数学计算"""
        expression = parameters.get("expression", "")
        if not expression:
            return "Error: No expression provided", ToolCallStatus.ERROR
        
        if self.fake_mode:
            logger.info(f"🧮 [FAKE] Executing calculator: {expression}")
            return "dummy calculator output", ToolCallStatus.SUCCESS
        
        try:
            # 简单的数学表达式计算
            safe_pattern = r'^[0-9+\-*/().\s]+$'
            if not re.match(safe_pattern, expression):
                return "Error: Invalid expression", ToolCallStatus.ERROR
            
            # 使用eval计算（在受控环境中）
            result = eval(expression)
            return str(result), ToolCallStatus.SUCCESS
            
        except Exception as e:
            return f"Error: {str(e)}", ToolCallStatus.ERROR
