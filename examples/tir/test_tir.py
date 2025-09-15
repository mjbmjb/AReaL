import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import TIR modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tir_workflow import TIRWorkflow
from tool_manager import ToolManager
from math_reward import MathRewardFunction


async def test_tool_manager():
    """测试工具管理器"""
    print("Testing ToolManager...")
    
    tool_manager = ToolManager(timeout=10)
    
    # 测试Python执行
    python_code = "print(2 + 3)"
    result = await tool_manager.execute_python(python_code)
    print(f"Python execution result: {result}")
    assert "5" in result, f"Expected '5' in result, got: {result}"
    
    # 测试计算器
    calc_expr = "2 * 3 + 4"
    result = await tool_manager.execute_calculator(calc_expr)
    print(f"Calculator result: {result}")
    assert result == "10", f"Expected '10', got: {result}"
    
    # 测试不安全代码
    unsafe_code = "import os; os.system('ls')"
    result = await tool_manager.execute_python(unsafe_code)
    print(f"Unsafe code result: {result}")
    assert "Error" in result, f"Expected error for unsafe code, got: {result}"
    
    tool_manager.cleanup()
    print("ToolManager tests passed!")


def test_math_reward():
    """测试数学奖励函数"""
    print("Testing MathRewardFunction...")
    
    reward_fn = MathRewardFunction()
    
    # 测试正确答案
    reward = reward_fn("", "The answer is 42", answer="42")
    assert reward == 1.0, f"Expected 1.0 for correct answer, got: {reward}"
    
    # 测试错误答案
    reward = reward_fn("", "The answer is 41", answer="42")
    assert reward == 0.0, f"Expected 0.0 for wrong answer, got: {reward}"
    
    # 测试部分匹配
    reward = reward_fn("", "The result is 42", answer="42")
    assert reward == 1.0, f"Expected 1.0 for partial match, got: {reward}"
    
    print("MathRewardFunction tests passed!")


def test_answer_extraction():
    """测试答案提取"""
    print("Testing answer extraction...")
    
    reward_fn = MathRewardFunction()
    
    test_cases = [
        ("The answer is 42", "42"),
        ("Final answer: 25", "25"),
        ("Result: 100", "100"),
        ("Solution: 3.14", "3.14"),
        ("= 17", "17"),
    ]
    
    for text, expected in test_cases:
        extracted = reward_fn._extract_answer(text)
        assert extracted == expected, f"Expected '{expected}', got: '{extracted}' for text: '{text}'"
    
    print("Answer extraction tests passed!")


async def test_tir_workflow():
    """测试TIR工作流（需要模拟engine）"""
    print("Testing TIRWorkflow...")
    
    # 这里我们只测试工作流的初始化，实际的推理需要真实的engine
    from transformers import AutoTokenizer
    
    # 使用一个简单的tokenizer进行测试
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tool_manager = ToolManager()
    reward_fn = MathRewardFunction()
    
    # 创建TIR工作流
    workflow = TIRWorkflow(
        reward_fn=reward_fn,
        gconfig=None,  # 这里简化，实际需要GenerationHyperparameters
        tokenizer=tokenizer,
        tool_manager=tool_manager,
        max_turns=3,
    )
    
    tool_manager.cleanup()
    print("TIRWorkflow tests passed!")


def test_data_loading():
    """测试数据加载"""
    print("Testing data loading...")
    
    data_file = Path(__file__).parent / "data" / "sample_math.jsonl"
    assert data_file.exists(), f"Data file not found: {data_file}"
    
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    assert len(data) > 0, "No data loaded"
    assert "messages" in data[0], "Missing 'messages' field"
    assert "answer" in data[0], "Missing 'answer' field"
    
    print(f"Loaded {len(data)} samples")
    print("Data loading tests passed!")


async def main():
    """运行所有测试"""
    print("Running TIR tests...")
    
    try:
        await test_tool_manager()
        test_math_reward()
        test_answer_extraction()
        await test_tir_workflow()
        test_data_loading()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
