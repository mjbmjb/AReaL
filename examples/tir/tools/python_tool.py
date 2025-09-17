import asyncio
import re
import sys
import tempfile
import os
from typing import Dict, Any, Tuple

from areal.utils import logging
from .base import BaseTool, ToolType, ToolDescription, ToolCallStatus

logger = logging.getLogger("Python Tool")


def extract_python_code(text: str) -> str:
    """从文本中提取Python代码，支持两种格式：
    1. ```python\n...\n```
    2. <python>...</python>
    
    Args:
        text: 包含Python代码的文本
        
    Returns:
        提取的Python代码，如果未找到则返回空字符串
    """
    # 尝试匹配 ```python``` 格式，从后往前只匹配最后一个
    pattern1 = r"```python\n(.*?)\n```"
    matches1 = list(re.finditer(pattern1, text, re.DOTALL | re.IGNORECASE))
    if matches1:
        last_match = matches1[-1]
        code = last_match.group(1).strip()
        logger.info(f"📝 Extracted Python code from ```python``` format (last occurrence): {code[:100]}...")
        return code
    
    # 尝试匹配 <python></python> 格式，从后往前只匹配最后一个
    pattern2 = r"<python>(.*?)</python>"
    matches2 = list(re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE))
    if matches2:
        last_match = matches2[-1]
        code = last_match.group(1).strip()
        logger.info(f"📝 Extracted Python code from <python> format (last occurrence): {code[:100]}...")
        return code
    
    logger.warning("⚠️ No Python code block found in either format")
    return ""


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
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>"
        )
    
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """从文本中提取Python代码，支持两种格式：```python``` 和 <python>"""
        code = extract_python_code(text)
        return {"code": code}
    
    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """执行Python代码"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR
        
        if self.fake_mode:
            logger.info(f"🐍 [FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output", ToolCallStatus.SUCCESS
        
        try:
            # 直接调用apply，避免在异步环境中使用ProcessPool
            result = self.python_executor.apply(code)
            logger.info(f"✅ Python execution completed: {str(result)[:100]}...")
            return str(result), ToolCallStatus.SUCCESS
        except Exception as e:
            logger.error(f"❌ Python execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR


class PythonTool(BaseTool):
    """Python代码执行工具（沙箱版本）"""
    
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
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n或者\n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>"
        )
    
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """从文本中提取Python代码，支持两种格式：```python``` 和 <python>"""
        code = extract_python_code(text)
        return {"code": code}
    
    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """执行Python代码"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR
        
        if self.fake_mode:
            logger.info(f"🐍 [FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output", ToolCallStatus.SUCCESS
        
        logger.info(f"🐍 Executing Python code: {code[:100]}...")
        
        try:
            # 安全检查
            if not self._is_safe_code(code):
                logger.warning("⚠️ Unsafe code detected, blocking execution")
                return "Error: Unsafe code detected", ToolCallStatus.ERROR
            
            # 在沙箱中执行
            result = asyncio.run(self._execute_in_sandbox(code))
            logger.info(f"✅ Python execution completed: {result[:100]}...")
            return result, ToolCallStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"❌ Python execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR
    
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
