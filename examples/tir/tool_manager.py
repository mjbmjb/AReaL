import asyncio
import re
import subprocess
import sys
from typing import Dict, Any, Optional
import tempfile
import os

from areal.utils import logging

logger = logging.getLogger("Tool Manager")


class ToolManager:
    """工具管理器，负责执行各种工具调用"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.sandbox_dir = tempfile.mkdtemp(prefix="tir_sandbox_")
        
    async def execute_python(self, code: str) -> str:
        """执行Python代码"""
        logger.info(f"🐍 Executing Python code: {code[:100]}...")
        
        try:
            # 安全检查
            logger.debug("🔒 Performing safety check")
            if not self._is_safe_code(code):
                logger.warning("⚠️ Unsafe code detected, blocking execution")
                return "Error: Unsafe code detected"
            
            logger.debug("✅ Code passed safety check")
            
            # 在沙箱中执行
            logger.debug("🏃 Starting code execution in sandbox")
            result = await self._execute_in_sandbox(code)
            logger.info(f"✅ Python execution completed: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Python execution error: {e}")
            return f"Error: {str(e)}"
    
    def _is_safe_code(self, code: str) -> bool:
        """检查代码是否安全"""
        # 禁止的危险操作
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
        logger.debug(f"📁 Creating temporary file in sandbox: {self.sandbox_dir}")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            dir=self.sandbox_dir, 
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        logger.debug(f"📝 Temporary file created: {temp_file}")
        
        try:
            # 执行Python代码
            logger.debug(f"🚀 Starting subprocess with timeout: {self.timeout}s")
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.sandbox_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            logger.debug(f"📊 Process completed with return code: {process.returncode}")
            logger.debug(f"📤 stdout: {stdout.decode('utf-8')[:200]}...")
            if stderr:
                logger.debug(f"⚠️ stderr: {stderr.decode('utf-8')[:200]}...")
            
            # 清理临时文件
            os.unlink(temp_file)
            logger.debug("🗑️ Temporary file cleaned up")
            
            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                logger.info(f"✅ Code execution successful: {result[:100]}...")
                return result
            else:
                error_msg = stderr.decode('utf-8').strip()
                logger.error(f"❌ Code execution failed: {error_msg}")
                return f"Error: {error_msg}"
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Code execution timeout after {self.timeout}s")
            return "Error: Execution timeout"
        except Exception as e:
            logger.error(f"💥 Unexpected error during execution: {e}")
            return f"Error: {str(e)}"
        finally:
            # 确保清理临时文件
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug("🗑️ Emergency cleanup of temporary file")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup temporary file: {cleanup_error}")
    
    async def execute_calculator(self, expression: str) -> str:
        """执行基础数学计算"""
        try:
            # 简单的数学表达式计算
            # 只允许基本的数学运算
            safe_pattern = r'^[0-9+\-*/().\s]+$'
            if not re.match(safe_pattern, expression):
                return "Error: Invalid expression"
            
            # 使用eval计算（在受控环境中）
            result = eval(expression)
            return str(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """清理沙箱目录"""
        try:
            import shutil
            if os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")


class PythonExecutor:
    """Python代码执行器"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    async def execute(self, code: str) -> str:
        """执行Python代码"""
        manager = ToolManager(timeout=self.timeout)
        return await manager.execute_python(code)


class Calculator:
    """基础计算器"""
    
    def __init__(self):
        pass
    
    async def execute(self, expression: str) -> str:
        """执行数学表达式"""
        manager = ToolManager()
        return await manager.execute_calculator(expression)
