import asyncio
import re
import subprocess
import sys
import traceback
from typing import Dict, Any, Optional, List, Union
import tempfile
import os
import json
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from areal.utils import logging

logger = logging.getLogger("Tool Manager")


class ToolType(Enum):
    """工具类型枚举"""
    PYTHON = "python"
    CALCULATOR = "calculator"


@dataclass
class ToolCall:
    """工具调用数据结构"""
    tool_type: ToolType
    parameters: Dict[str, Any]
    raw_text: str


@dataclass
class ToolDescription:
    """工具描述数据结构"""
    name: str
    description: str
    parameters: Dict[str, str]  # 参数名 -> 参数描述
    parameter_prompt: str  # 参数解析的prompt
    example: str


class BaseTool(ABC):
    """基础工具抽象类"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.timeout = timeout
        self.fake_mode = fake_mode
    
    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        """工具类型"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> ToolDescription:
        """工具描述"""
        pass
    
    @abstractmethod
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """解析参数"""
        pass
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass


class QwenPythonTool(BaseTool):
    """Qwen Python代码执行工具"""

    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        super().__init__(timeout, fake_mode)
        from qwen_agent.tools.python_executor import PythonExecutor
        self.python_executor = PythonExecutor()

    @property
    def tool_type(self) -> ToolType:
        return ToolType.PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python_executor",
            description="Execute Python code. Supports variable calculation, data processing, algorithm implementation, etc.",
            parameters={
                "code": "The Python code string to execute"
            },
            parameter_prompt="Please provide the Python code to execute. Supports variable calculation, data processing, algorithm implementation, etc.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```"
        )
    
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """从```python```标记中提取Python代码"""
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            code = match.group(1).strip()
            logger.info(f"📝 Extracted Python code: {code[:100]}...")
            return {"code": code}
        else:
            logger.warning("⚠️ No ```python``` tag found")
            return {"code": ""}
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        """执行Python代码"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", False
        
        if self.fake_mode:
            logger.info(f"🐍 [FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output", True
        
        try:
            # 直接调用apply，避免在异步环境中使用ProcessPool
            result = self.python_executor.apply(code)
            logger.info(f"✅ Python execution completed: {str(result)[:100]}...")
            return str(result), True
        except Exception as e:
            logger.error(f"❌ Python execution error: {e}")
            return f"Error: {str(e)}", False
   

class PythonTool(BaseTool):
    """Python代码执行工具"""
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.PYTHON
    
    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python_executor",
            description="执行Python代码，支持变量计算、数据处理、算法实现等",
            parameters={
                "code": "要执行的Python代码字符串"
            },
            parameter_prompt="请提供要执行的Python代码，支持变量计算、数据处理、算法实现等",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```"
        )
    
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """从```python```标记中提取Python代码"""
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            code = match.group(1).strip()
            logger.info(f"📝 Extracted Python code: {code[:100]}...")
            return {"code": code}
        else:
            logger.warning("⚠️ No ```python``` tag found")
            return {"code": ""}
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        """执行Python代码"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided"
        
        if self.fake_mode:
            logger.info(f"🐍 [FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output"
        
        logger.info(f"🐍 Executing Python code: {code[:100]}...")
        
        try:
            # 安全检查
            if not self._is_safe_code(code):
                logger.warning("⚠️ Unsafe code detected, blocking execution")
                return "Error: Unsafe code detected"
            
            # 在沙箱中执行
            result = self._execute_in_sandbox(code)
            logger.info(f"✅ Python execution completed: {result[:100]}...")
            return result, True
            
        except Exception as e:
            logger.error(f"❌ Python execution error: {e}")
            return f"Error: {str(e)}", False
    
    def _is_safe_code(self, code: str) -> bool:
        """检查代码是否安全"""
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"__import__",
            r"exec\s*\(",
            r"eval\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"exit\s*\(",
            r"quit\s*\(",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True
    
    async def _execute_in_sandbox(self, code: str) -> str:
        """在沙箱中执行Python代码"""
        sandbox_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                dir=sandbox_dir, 
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            # 执行Python代码
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                return result
            else:
                error_msg = stderr.decode('utf-8').strip()
                return f"Error: {error_msg}"
                
        except asyncio.TimeoutError:
            return "Error: Execution timeout"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # 清理沙箱目录
            try:
                import shutil
                if os.path.exists(sandbox_dir):
                    shutil.rmtree(sandbox_dir)
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup sandbox: {cleanup_error}")


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
    
    def execute(self, parameters: Dict[str, Any]) -> str:
        """执行数学计算"""
        expression = parameters.get("expression", "")
        if not expression:
            return "Error: No expression provided", False
        
        if self.fake_mode:
            logger.info(f"🧮 [FAKE] Executing calculator: {expression}")
            return "dummy calculator output", True
        
        try:
            # 简单的数学表达式计算
            safe_pattern = r'^[0-9+\-*/().\s]+$'
            if not re.match(safe_pattern, expression):
                return "Error: Invalid expression", False
            
            # 使用eval计算（在受控环境中）
            result = eval(expression)
            return str(result), True
            
        except Exception as e:
            return f"Error: {str(e)}", False


class ToolRegistry:
    """工具注册表，管理所有可用工具"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.tools = {
            ToolType.PYTHON: QwenPythonTool(timeout, fake_mode),
            ToolType.CALCULATOR: CalculatorTool(timeout, fake_mode),
        }
        # 工具标记映射
        self.tool_markers = {
            ToolType.PYTHON: ("```python\n", "\n```"),
            ToolType.CALCULATOR: ("<calculator>", "</calculator>"),
        }
    
    def get_tool(self, tool_type: ToolType) -> Optional[BaseTool]:
        """获取工具实例"""
        return self.tools.get(tool_type)
    
    def get_all_tools(self) -> Dict[ToolType, BaseTool]:
        """获取所有工具实例"""
        return self.tools
    
    def get_tool_markers(self) -> Dict[ToolType, tuple[str, str]]:
        """获取所有工具的标记信息
        
        Returns:
            Dict[ToolType, tuple[str, str]]: 工具类型 -> (开始标记, 结束标记)
        """
        return self.tool_markers.copy()
    
    def get_all_start_markers(self) -> List[str]:
        """获取所有开始标记
        
        Returns:
            List[str]: 所有开始标记的列表
        """
        return [markers[0] for markers in self.tool_markers.values()]
    
    def get_all_end_markers(self) -> List[str]:
        """获取所有结束标记
        
        Returns:
            List[str]: 所有结束标记的列表
        """
        return [markers[1] for markers in self.tool_markers.values()]
    
    def get_all_markers(self) -> List[str]:
        """获取所有标记（开始和结束）
        
        Returns:
            List[str]: 所有标记的列表
        """
        all_markers = []
        for start_marker, end_marker in self.tool_markers.values():
            all_markers.extend([start_marker, end_marker])
        return all_markers
    
    def get_tool_descriptions_prompt(self) -> str:
        """生成工具描述的prompt文本，供外部调用"""
        prompt_parts = ["可用工具列表：\n"]
        
        for tool_type, tool in self.tools.items():
            desc = tool.description
            prompt_parts.append(f"Tool Name: {desc.name}")
            prompt_parts.append(f"Description: {desc.description}")
            prompt_parts.append(f"Parameter Description: {desc.parameter_prompt}")
            prompt_parts.append(f"Usage Example: {desc.example}")
            prompt_parts.append("---")
        
        return "\n".join(prompt_parts)


class ToolRouter:
    """工具路由器，根据标记判断需要调用哪个工具"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_markers = {
            ToolType.PYTHON: r"```python\n(.*?)\n```",
            ToolType.CALCULATOR: r"<calculator>(.*?)</calculator>",
        }
    
    def route(self, text: str) -> Optional[ToolType]:
        """根据标记判断需要调用的工具类型"""
        text = text.strip()
        
        # 检查每个工具的标记
        for tool_type, pattern in self.tool_markers.items():
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                logger.info(f"🔀 Routed to {tool_type.value} based on marker")
                return tool_type
        
        logger.warning(f"⚠️ No tool marker found in text: {text[-50:]}...")
        return None


class FakeToolManager:
    """假的工具管理器，用于调试，直接返回dummy结果"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        logger.info("🎭 Initializing FakeToolManager for debugging")
        
    async def execute_python(self, code: str) -> str:
        """执行Python代码 - 假版本"""
        logger.info(f"🐍 [FAKE] Executing Python code: {code[:100]}...")
        return "dummy code output"
    
    async def execute_calculator(self, expression: str) -> str:
        """执行基础数学计算 - 假版本"""
        logger.info(f"🧮 [FAKE] Executing calculator: {expression}")
        return "dummy calculator output"
    
    def cleanup(self):
        """清理 - 假版本"""
        logger.info("🗑️ [FAKE] Cleanup completed")


class ToolManager:
    """通用工具管理器，负责协调工具调用"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.timeout = timeout
        self.fake_mode = fake_mode
        self.registry = ToolRegistry(timeout, fake_mode)
        self.router = ToolRouter(self.registry)
        
        logger.info(f"🔧 Initialized ToolManager (fake_mode={fake_mode})")
    
    def get_tool_descriptions_prompt(self) -> str:
        """获取工具描述的prompt文本，供外部调用"""
        return self.registry.get_tool_descriptions_prompt()
    
    def get_tool_markers(self) -> Dict[ToolType, tuple[str, str]]:
        """获取所有工具的标记信息
        
        Returns:
            Dict[ToolType, tuple[str, str]]: 工具类型 -> (开始标记, 结束标记)
        """
        return self.registry.get_tool_markers()
    
    def get_all_start_markers(self) -> List[str]:
        """获取所有开始标记，用于设置stop token
        
        Returns:
            List[str]: 所有开始标记的列表，如 ['```python\n', '<calculator>']
        """
        return self.registry.get_all_start_markers()
    
    def get_all_end_markers(self) -> List[str]:
        """获取所有结束标记，用于设置stop token
        
        Returns:
            List[str]: 所有结束标记的列表，如 ['\n```', '</calculator>']
        """
        return self.registry.get_all_end_markers()
    
    def get_all_markers(self) -> List[str]:
        """获取所有标记（开始和结束），用于设置stop token
        
        Returns:
            List[str]: 所有标记的列表，如 ['```python\n', '\n```', '<calculator>', '</calculator>']
        """
        return self.registry.get_all_markers()
    
    def execute_tool_call(self, text: str) -> str:
        """统一的工具调用接口"""
        logger.info(f"🔧 Processing tool call: {text[-100:]}...")
        
        # 1. 路由：判断需要调用哪个工具
        tool_type = self.router.route(text)
        if not tool_type:
            return "Error: No suitable tool found for the given text", False
        
        # 2. 获取工具实例
        tool = self.registry.get_tool(tool_type)
        if not tool:
            return f"Error: Tool {tool_type.value} not found", False
        
        # 3. 解析参数
        try:
            parameters = tool.parse_parameters(text)
            logger.info(f"📋 Parsed parameters: {parameters}")
        except Exception as e:
            logger.error(f"❌ Parameter parsing error: {e}")
            return f"Error: Failed to parse parameters - {str(e)}", False
        
        # 4. 执行工具
        result, status = tool.execute(parameters)
        if status == True:
            logger.info(f"✅ Tool execution completed: {result}")
            return result, status
        else:
            logger.error(f"❌ Tool execution error: {result}")
            return f"Error: Tool execution failed - {result}", status
    
    def cleanup(self):
        """清理资源"""
        logger.info("🗑️ ToolManager cleanup completed")


# 使用示例
async def main():
    """使用示例"""
    manager = ToolManager(fake_mode=False)  # 使用假模式进行测试
    
    # 获取工具描述
    print("=== 工具描述 ===")
    print(manager.get_tool_descriptions_prompt())
    print()
    
    # 获取标记信息（用于设置stop token）
    print("=== 工具标记信息 ===")
    print("所有工具标记:", manager.get_tool_markers())
    print("开始标记:", manager.get_all_start_markers())
    print("结束标记:", manager.get_all_end_markers())
    print("所有标记:", manager.get_all_markers())
    print()
    
    # 测试工具调用
    test_cases = [
        "<calculator>1 + 2 * 3</calculator>",
        "```python\nprint('Hello World')\n```",
        "```python\nfor i in range(3):\n    print(i)\n```",
        "<calculator>(10 + 5) / 3</calculator>",
    ]
    
    print("=== 工具调用测试 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case}")
        result = await manager.execute_tool_call(test_case)
        print(f"结果: {result}")
    
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
