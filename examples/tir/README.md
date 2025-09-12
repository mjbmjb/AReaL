# Tool-Integrated Reasoning (TIR) Agent

这是一个在AReaL框架中实现的Tool-Integrated Reasoning智能体，能够在数学推理过程中通过多轮工具调用来解决复杂问题。

## 项目结构

```
examples/tir/
├── README.md                    # 项目说明文档
├── TIR_IMPLEMENTATION_DOC.md   # 详细实现文档
├── tir_workflow.py             # 核心工作流实现
├── tool_manager.py             # 工具管理器
├── math_reward.py              # 数学奖励函数
├── tir_config.yaml             # 配置文件
├── train_tir.py                # 训练脚本
├── test_tir.py                 # 测试脚本
├── tools/                      # 工具实现
│   └── __init__.py
├── data/                       # 数据文件
│   └── sample_math.jsonl       # 示例数据
└── utils/                      # 工具函数
    └── __init__.py
```

## 核心特性

### 1. 多轮工具调用
- 支持在推理过程中调用Python代码执行器
- 支持基础数学计算器
- 智能检测何时需要使用工具

### 2. 流式生成与工具调用检测
- 使用特殊token机制检测工具调用意图
- 在生成过程中实时检测并执行工具
- 支持多轮对话和上下文管理

### 3. 安全的工具执行
- 沙箱环境执行Python代码
- 代码安全检查，防止恶意操作
- 执行超时和资源限制

### 4. 数学奖励函数
- 智能提取最终答案
- 支持多种答案格式
- 数学等价性检查

## 快速开始

### 1. 安装依赖

确保已安装AReaL框架及其依赖。

### 2. 运行测试

```bash
cd examples/tir
python test_tir.py
```

### 3. 训练模型

```bash
# 单节点训练
python3 -m areal.launcher.local \
  examples/tir/train_tir.py \
  --config examples/tir/tir_config.yaml

# 多节点训练
python3 -m areal.launcher.ray \
  examples/tir/train_tir.py \
  --config examples/tir/tir_config.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8
```

## 核心组件

### TIRWorkflow
继承自AReaL的`RolloutWorkflow`，实现多轮工具调用的推理过程。

**主要功能**:
- 管理多轮对话状态
- 检测工具调用意图
- 执行工具并整合结果
- 计算奖励分数

### ToolManager
工具管理器，负责执行各种工具调用。

**支持的工具**:
- **Python执行器**: 执行Python代码进行数学计算
- **计算器**: 基础数学运算
- **安全沙箱**: 确保代码执行安全

### MathRewardFunction
数学推理奖励函数，评估推理的正确性。

**特性**:
- 智能答案提取
- 多种答案格式支持
- 数学等价性检查

## 配置说明

### 模型配置
- **基础模型**: Qwen2.5-1.5B-Instruct
- **数据集**: inclusionAI/AReaL-boba-Data
- **训练算法**: PPO

### TIR特定配置
```yaml
tir:
  max_turns: 5                    # 最大推理轮数
  tool_timeout: 30                # 工具执行超时
  enable_tools: ["python", "calculator"]  # 启用的工具
```

## 使用示例

### 基本使用
```python
from tir_workflow import TIRWorkflow
from tool_manager import ToolManager
from math_reward import MathRewardFunction

# 初始化组件
tool_manager = ToolManager()
reward_fn = MathRewardFunction()

# 创建工作流
workflow = TIRWorkflow(
    reward_fn=reward_fn,
    gconfig=gconfig,
    tokenizer=tokenizer,
    tool_manager=tool_manager,
    max_turns=5
)

# 运行推理
result = await workflow.arun_episode(engine, data)
```

### 工具调用示例
```python
# Python代码执行
python_code = "print(2 + 3 * 4)"
result = await tool_manager.execute_python(python_code)
# 输出: 14

# 数学计算
calc_expr = "sqrt(144) + 2**3"
result = await tool_manager.execute_calculator(calc_expr)
# 输出: 20
```

## 训练效果

### 预期指标
- **工具调用准确率**: 智能体正确识别需要使用工具的情况
- **数学问题准确率**: 在数学数据集上的最终答案正确率
- **训练收敛性**: 训练过程中的奖励曲线收敛情况

### 监控指标
- 奖励分数变化
- 工具调用频率
- 推理轮数分布
- 答案正确率

## 扩展性

### 添加新工具
1. 在`ToolManager`中注册新工具
2. 实现工具执行逻辑
3. 更新工具调用检测模式

### 支持新领域
1. 实现领域特定的奖励函数
2. 添加领域相关的工具
3. 调整推理策略

## 注意事项

1. **安全性**: 工具执行在沙箱环境中进行，但仍需注意代码安全
2. **性能**: 多轮推理会增加计算开销，建议合理设置最大轮数
3. **内存**: 长对话历史可能消耗大量内存，考虑实现上下文压缩

## 故障排除

### 常见问题
1. **工具执行失败**: 检查代码安全性和语法
2. **奖励计算错误**: 验证答案提取模式
3. **训练不收敛**: 调整学习率和奖励缩放

### 调试技巧
1. 启用详细日志记录
2. 检查生成的对话历史
3. 监控工具执行结果

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

遵循AReaL项目的Apache 2.0许可证。
