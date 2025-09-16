from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from areal.utils import logging

logger = logging.getLogger("Tool Base")


class ToolCallStatus(Enum):
    """工具调用状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


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
    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """执行工具
        
        Returns:
            Tuple[str, ToolCallStatus]: (结果, 状态)
        """
        pass
