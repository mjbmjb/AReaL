import asyncio
import re
from typing import Dict, Any, Optional, List, Tuple

from areal.utils import logging
from .tools import (
    ToolCallStatus,
    ToolType,
    ToolCall,
    ToolDescription,
    BaseTool,
    QwenPythonTool,
    CalculatorTool,
    extract_python_code,
)

logger = logging.getLogger("Tool Manager")


class ToolRegistry:
    """工具注册表，管理所有可用工具"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.tools = {
            ToolType.PYTHON: QwenPythonTool(timeout, fake_mode),
            ToolType.CALCULATOR: CalculatorTool(timeout, fake_mode),
        }
        # 工具标记映射 - 分别定义开始和结束标记
        self.tool_start_markers = {
            ToolType.PYTHON: ["```python\n", "<python>"],
            ToolType.CALCULATOR: ["<calculator>"],
        }
        self.tool_end_markers = {
            ToolType.PYTHON: ["\n```", "</python>"],
            ToolType.CALCULATOR: ["</calculator>"],
        }
    
    def get_tool(self, tool_type: ToolType) -> Optional[BaseTool]:
        """获取工具实例"""
        return self.tools.get(tool_type)
    
    def get_all_tools(self) -> Dict[ToolType, BaseTool]:
        """获取所有工具实例"""
        return self.tools
    
    def get_tool_markers(self) -> Dict[ToolType, Tuple[List[str], List[str]]]:
        """获取所有工具的标记信息
        
        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: 工具类型 -> (开始标记列表, 结束标记列表)
        """
        return {
            tool_type: (self.tool_start_markers[tool_type], self.tool_end_markers[tool_type])
            for tool_type in self.tool_start_markers.keys()
        }
    
    def get_all_start_markers(self) -> List[str]:
        """获取所有开始标记
        
        Returns:
            List[str]: 所有开始标记的列表
        """
        start_markers = []
        for markers in self.tool_start_markers.values():
            start_markers.extend(markers)
        return start_markers
    
    def get_all_end_markers(self) -> List[str]:
        """获取所有结束标记
        
        Returns:
            List[str]: 所有结束标记的列表
        """
        end_markers = []
        for markers in self.tool_end_markers.values():
            end_markers.extend(markers)
        return end_markers
    
    def get_all_markers(self) -> List[str]:
        """获取所有标记（开始和结束）
        
        Returns:
            List[str]: 所有标记的列表
        """
        all_markers = []
        all_markers.extend(self.get_all_start_markers())
        all_markers.extend(self.get_all_end_markers())
        return all_markers
    
    def get_tool_descriptions_prompt(self) -> str:
        """生成工具描述的prompt文本，供外部调用"""
        prompt_parts = ["Tools List:\n"]
        
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
        self.tool_markers = [
            (ToolType.PYTHON, r"```python\n(.*?)\n```"),
            (ToolType.PYTHON, r"<python>(.*?)</python>"),
            (ToolType.CALCULATOR, r"<calculator>(.*?)</calculator>"),
        ]
    
    def route(self, text: str) -> Optional[ToolType]:
        """根据标记判断需要调用的工具类型"""
        text = text.strip()
        
        # 检查每个工具的标记
        for tool_type, pattern in self.tool_markers:
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
    
    def get_tool_markers(self) -> Dict[ToolType, Tuple[List[str], List[str]]]:
        """获取所有工具的标记信息
        
        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: 工具类型 -> (开始标记列表, 结束标记列表)
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
    
    def execute_tool_call(self, text: str) -> Tuple[str, ToolCallStatus]:
        """统一的工具调用接口
        
        Returns:
            Tuple[str, ToolCallStatus]: (结果, 状态)
        """
        logger.info(f"🔧 Processing tool call: {text[-100:]}...")
        
        # 1. 路由：判断需要调用哪个工具
        tool_type = self.router.route(text)
        if not tool_type:
            return "Error: No suitable tool found for the given text", ToolCallStatus.NOT_FOUND
        
        # 2. 获取工具实例
        tool = self.registry.get_tool(tool_type)
        if not tool:
            return f"Error: Tool {tool_type.value} not found", ToolCallStatus.NOT_FOUND
        
        # 3. 解析参数
        try:
            parameters = tool.parse_parameters(text)
            logger.info(f"📋 Parsed parameters: {parameters}")
        except Exception as e:
            logger.error(f"❌ Parameter parsing error: {e}")
            return f"Error: Failed to parse parameters - {str(e)}", ToolCallStatus.ERROR
        
        # 4. 执行工具
        result, status = tool.execute(parameters)
        if status == ToolCallStatus.SUCCESS:
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
        "<python>print('Hello from <python> tag')\nfor i in range(2):\n    print(f'Count: {i}')\n</python>",
        "<calculator>(10 + 5) / 3</calculator>",
    ]
    
    print("=== 工具调用测试 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case}")
        result, status = manager.execute_tool_call(test_case)
        print(f"结果: {result}")
        print(f"状态: {status}")
    
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())