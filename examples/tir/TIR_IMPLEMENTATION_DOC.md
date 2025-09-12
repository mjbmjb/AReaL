# Tool-Integrated Reasoning (TIR) Agent 实现文档

## 项目概述

本项目旨在在AReaL框架中实现一个Tool-Integrated Reasoning智能体，该智能体能够在数学推理过程中通过多轮工具调用来解决复杂问题。智能体将使用Python代码执行、数学计算等工具，并通过强化学习进行端到端训练。

## 核心设计思路

### 1. 整体架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Math Problem  │───▶│  TIR Workflow    │───▶│  Tool Execution │
│   (Input)       │    │  (Multi-turn)    │    │  (Python/Calc)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Reward Function │
                       │  (Math Accuracy) │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  PPO Training    │
                       │  (AReaL Engine)  │
                       └──────────────────┘
```

### 2. 关键技术挑战与解决方案

#### 2.1 流式生成中的工具调用检测

**挑战**: 在生成过程中实时检测工具调用意图，不能等到完整序列生成后再检测。

**解决方案**: 
- 使用特殊的stop token机制，当模型生成到特定token时暂停生成
- 检查当前生成内容是否包含工具调用模式
- 如果包含工具调用，执行工具并继续生成；否则继续正常生成

#### 2.2 多轮对话的状态管理

**挑战**: 维护对话历史和工具执行上下文，确保推理的连贯性。

**解决方案**:
- 使用状态机模式管理对话状态
- 维护完整的对话历史，包括工具调用和结果
- 实现上下文压缩机制，避免序列过长

#### 2.3 工具执行的安全性

**挑战**: 防止恶意代码执行，确保系统安全。

**解决方案**:
- 使用沙箱环境执行Python代码
- 实现代码审查和过滤机制
- 设置执行超时和资源限制

## 详细技术实现

### 3. 核心组件设计

#### 3.1 TIRWorkflow (核心工作流)

```python
class TIRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        tool_manager: ToolManager,
        max_turns: int = 5,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
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
        self.tool_call_tokens = self._setup_tool_tokens()
```

**关键特性**:
- 继承自AReaL的`RolloutWorkflow`基类
- 支持多轮工具调用的推理过程
- 实现流式生成和工具调用检测
- 集成奖励函数计算

#### 3.2 工具调用机制设计

**特殊Token设计**:
```python
# 工具调用相关token
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"
TOOL_PYTHON_START = "<python>"
TOOL_PYTHON_END = "</python>"
```

**流式生成与工具调用检测**:
```python
async def arun_episode(self, engine: InferenceEngine, data):
    # 1. 初始化对话历史
    messages = data["messages"]
    conversation_history = []
    
    # 2. 多轮推理循环
    for turn in range(self.max_turns):
        # 2.1 生成响应（包含可能的工具调用）
        response = await self._generate_with_tool_detection(engine, messages)
        
        # 2.2 检测并执行工具调用
        if self._has_tool_call(response):
            tool_results = await self._execute_tools(response)
            # 2.3 将工具结果整合到对话中
            messages = self._integrate_tool_results(messages, tool_results)
        else:
            # 2.4 没有工具调用，检查是否完成
            if self._is_final_answer(response):
                break
                
    # 3. 计算奖励
    reward = self._calculate_reward(conversation_history, data["answer"])
    return self._format_trajectory(conversation_history, reward)
```

#### 3.3 ToolManager (工具管理器)

```python
class ToolManager:
    def __init__(self):
        self.tools = {
            "python": PythonExecutor(),
            "calculator": Calculator(),
            "sympy": SympyExecutor()
        }
        self.sandbox = CodeSandbox()  # 安全沙箱
    
    async def execute_tool(self, tool_name: str, code: str) -> str:
        """异步执行工具并返回结果"""
        if tool_name not in self.tools:
            return f"Error: Unknown tool {tool_name}"
        
        try:
            # 在沙箱中执行代码
            result = await self.sandbox.execute(
                self.tools[tool_name], 
                code,
                timeout=30
            )
            return f"<tool_result>{result}</tool_result>"
        except Exception as e:
            return f"<tool_result>Error: {str(e)}</tool_result>"
```

**支持的工具**:
- **Python执行器**: 执行Python代码进行数学计算
- **计算器**: 基础数学运算
- **符号计算**: 使用SymPy进行符号数学计算

#### 3.4 MathRewardFunction (奖励函数)

```python
class MathRewardFunction:
    def __init__(self):
        self.parser = MathAnswerParser()
        self.sympy_parser = SympyParser()
    
    def __call__(self, prompt, completions, **kwargs):
        # 1. 提取最终答案
        final_answer = self._extract_final_answer(completions)
        ground_truth = kwargs.get("answer", "")
        
        # 2. 数学等价性检查
        if self._math_equal(final_answer, ground_truth):
            return 1.0  # 正确
        else:
            return 0.0  # 错误
    
    def _math_equal(self, answer1: str, answer2: str) -> bool:
        """检查两个数学表达式是否等价"""
        try:
            # 使用SymPy进行符号等价性检查
            expr1 = self.sympy_parser.parse(answer1)
            expr2 = self.sympy_parser.parse(answer2)
            return sp.simplify(expr1 - expr2) == 0
        except:
            # 回退到数值比较
            try:
                val1 = float(answer1)
                val2 = float(answer2)
                return abs(val1 - val2) < 1e-6
            except:
                return False
```

### 4. 训练配置设计

#### 4.1 模型和数据选择

**基础模型**: 
- 推荐使用 `Qwen2.5-1.5B-Instruct` 或 `Qwen2.5-7B-Instruct`
- 支持工具调用和数学推理能力

**数据集**:
- 主要使用 `inclusionAI/AReaL-boba-Data` 数学数据集
- 可以添加自定义的数学推理数据集

#### 4.2 超参数配置

```yaml
# tir_config.yaml
experiment_name: tir-math-reasoning
trial_name: trial0
allocation_mode: sglang.d4p1t1+d4p1t1

# 模型配置
actor:
  path: Qwen/Qwen2.5-1.5B-Instruct
  dtype: bfloat16
  max_new_tokens: 2048

# 生成配置
gconfig:
  n_samples: 4
  max_new_tokens: 1024
  temperature: 1.0
  stop_token_ids: [TOOL_CALL_START_TOKEN_ID]  # 工具调用停止token

# TIR特定配置
tir:
  max_turns: 5
  tool_timeout: 30
  enable_tools: ["python", "calculator"]
  tool_call_tokens:
    start: "<tool_call>"
    end: "</tool_call>"
    python_start: "<python>"
    python_end: "</python>"
```

### 5. 实现步骤规划

#### Phase 1: 基础框架搭建 (1-2天)
1. 创建 `examples/tir/` 目录结构
2. 实现 `TIRWorkflow` 基础类
3. 实现 `ToolManager` 工具管理器
4. 创建基础配置文件

#### Phase 2: 工具集成 (2-3天)
1. 实现Python代码执行器
2. 实现数学计算器
3. 实现工具调用解析器
4. 测试工具执行流程

#### Phase 3: 工作流完善 (2-3天)
1. 完善多轮对话逻辑
2. 实现流式生成和工具调用检测
3. 集成奖励函数
4. 端到端测试

#### Phase 4: 训练和优化 (3-4天)
1. 配置训练环境
2. 运行训练实验
3. 监控训练指标
4. 调优超参数

#### Phase 5: 文档和展示 (1-2天)
1. 编写README文档
2. 创建使用示例
3. 准备Pull Request

### 6. 文件结构设计

```
examples/tir/
├── README.md                    # 项目说明文档
├── tir_workflow.py             # 核心工作流实现
├── tool_manager.py             # 工具管理器
├── math_reward.py              # 数学奖励函数
├── tir_config.yaml             # 配置文件
├── train_tir.py                # 训练脚本
├── eval_tir.py                 # 评估脚本
├── tools/                      # 工具实现
│   ├── __init__.py
│   ├── python_executor.py      # Python执行器
│   ├── calculator.py           # 计算器
│   ├── sympy_executor.py       # 符号计算
│   └── sandbox.py              # 安全沙箱
├── data/                       # 数据文件
│   ├── sample_math.jsonl       # 示例数据
│   └── test_cases.jsonl        # 测试用例
└── utils/                      # 工具函数
    ├── __init__.py
    ├── math_parser.py          # 数学表达式解析
    └── conversation.py         # 对话管理
```

### 7. 关键技术实现细节

#### 7.1 流式生成中的工具调用检测

```python
async def _generate_with_tool_detection(self, engine, messages):
    """带工具调用检测的生成"""
    # 1. 设置特殊的stop token
    gconfig = self.gconfig.new(
        stop_token_ids=[self.tool_call_start_token_id]
    )
    
    # 2. 生成到工具调用token
    response = await engine.agenerate(ModelRequest(
        input_ids=input_ids,
        gconfig=gconfig,
        tokenizer=self.tokenizer
    ))
    
    # 3. 检查是否包含工具调用
    if self._has_tool_call(response.output_tokens):
        # 4. 继续生成工具调用内容
        tool_response = await self._generate_tool_call(engine, response)
        return self._merge_responses(response, tool_response)
    
    return response
```

#### 7.2 工具执行的安全沙箱

```python
class CodeSandbox:
    def __init__(self):
        self.allowed_modules = ['math', 'numpy', 'sympy']
        self.max_execution_time = 30
        self.max_memory = 100 * 1024 * 1024  # 100MB
    
    async def execute(self, tool, code: str) -> str:
        """在安全沙箱中执行代码"""
        # 1. 代码安全检查
        if not self._is_safe_code(code):
            raise SecurityError("Unsafe code detected")
        
        # 2. 设置执行环境
        env = self._create_safe_environment()
        
        # 3. 异步执行
        result = await asyncio.wait_for(
            self._execute_in_sandbox(tool, code, env),
            timeout=self.max_execution_time
        )
        
        return result
```

### 8. 预期效果和评估指标

#### 8.1 功能指标
- **工具调用准确率**: 智能体正确识别需要使用工具的情况
- **工具执行成功率**: 工具调用能够成功执行并返回正确结果
- **多轮推理能力**: 能够通过多轮工具调用解决复杂问题

#### 8.2 性能指标
- **数学问题准确率**: 在数学数据集上的最终答案正确率
- **训练收敛性**: 训练过程中的奖励曲线和损失函数收敛情况
- **推理效率**: 平均每个问题需要的工具调用次数

#### 8.3 安全性指标
- **代码执行安全性**: 无恶意代码执行
- **资源使用控制**: 内存和CPU使用在合理范围内
- **执行超时控制**: 工具执行不会无限期阻塞

### 9. 扩展性设计

#### 9.1 工具扩展
- 支持添加新的工具类型（如Wolfram Alpha、Mathematica等）
- 工具注册机制，便于动态加载
- 工具优先级和选择策略

#### 9.2 领域扩展
- 支持其他推理任务（代码生成、逻辑推理等）
- 可配置的奖励函数
- 领域特定的工具集

#### 9.3 模型扩展
- 支持不同规模的模型
- 支持不同的模型架构
- 支持多模态输入（图像、文本等）

### 10. 风险评估和缓解策略

#### 10.1 技术风险
- **工具调用机制复杂性**: 通过模块化设计和充分测试缓解
- **多轮推理状态管理**: 使用状态机模式和清晰的接口设计
- **流式生成中断机制**: 参考AReaL现有实现，确保兼容性

#### 10.2 性能风险
- **多轮推理计算开销**: 通过异步执行和缓存机制优化
- **工具执行延迟**: 设置合理的超时和重试机制
- **内存使用增长**: 实现上下文压缩和清理机制

#### 10.3 安全风险
- **代码执行安全**: 使用沙箱环境和代码审查
- **资源滥用**: 设置严格的资源限制和监控
- **恶意输入**: 实现输入验证和过滤机制

## 总结

本实现文档详细规划了在AReaL框架中实现Tool-Integrated Reasoning智能体的完整方案。通过模块化设计、安全沙箱、流式生成等技术手段，我们能够构建一个既强大又安全的数学推理智能体。该方案充分考虑了AReaL框架的特点，确保与现有系统的良好集成，同时为未来的扩展留下了充分的空间。

关键成功因素：
1. **与AReaL框架的深度集成**: 充分利用现有的RL训练基础设施
2. **安全的工具执行机制**: 确保系统安全性的同时提供强大的计算能力
3. **流式生成和工具调用的无缝结合**: 实现真正的交互式推理
4. **模块化和可扩展的设计**: 便于后续功能扩展和维护
