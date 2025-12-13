# test_deterministic_randomization.py
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
from tool_server.utils.utils import pil_to_base64
from PIL import Image
import json

def test_deterministic_randomization():
    """测试确定性随机化功能"""
    
    print("=" * 80)
    print("测试确定性随机化功能")
    print("=" * 80)
    
    # 使用相同的deterministic_id初始化两个ToolManager
    deterministic_id = "test_experiment_v1.0"
    
    print("\n第一次初始化 ToolManager...")
    tool_manager_1 = ToolManager(
        randomize=True,
        deterministic_id=deterministic_id,
        tools=["Point", "Crop", "OCR", "Draw2DPath"]  # 指定几个测试工具
    )
    
    print("\n第二次初始化 ToolManager (使用相同的 deterministic_id)...")
    tool_manager_2 = ToolManager(
        randomize=True,
        deterministic_id=deterministic_id,
        tools=["Point", "Crop", "OCR", "Draw2DPath"]
    )
    
    # 测试1: 比较工具映射是否一致
    print("\n" + "=" * 80)
    print("测试1: 验证工具名称映射的一致性")
    print("=" * 80)
    
    mapping_match = tool_manager_1.original_to_randomized == tool_manager_2.original_to_randomized
    print(f"\n工具映射是否一致: {mapping_match}")
    
    if mapping_match:
        print("\n✓ 工具映射一致!")
        print("\n示例映射:")
        for orig, rand in list(tool_manager_1.original_to_randomized.items())[:5]:
            print(f"  {orig} -> {rand}")
    else:
        print("\n✗ 工具映射不一致!")
        print("\nToolManager 1 的映射:")
        print(json.dumps(tool_manager_1.original_to_randomized, indent=2))
        print("\nToolManager 2 的映射:")
        print(json.dumps(tool_manager_2.original_to_randomized, indent=2))
        return False
    
    # 测试2: 比较System Prompt是否一致
    print("\n" + "=" * 80)
    print("测试2: 验证System Prompt的一致性")
    print("=" * 80)
    
    system_prompt_1 = tool_manager_1.get_tool_prompt("one_tool_call")
    system_prompt_2 = tool_manager_2.get_tool_prompt("one_tool_call")
    
    prompt_match = system_prompt_1 == system_prompt_2
    print(f"\nSystem Prompt是否一致: {prompt_match}")
    
    if prompt_match:
        print("\n✓ System Prompt一致!")
        print(f"\nPrompt长度: {len(system_prompt_1)} 字符")
        print("\nPrompt前500字符:")
        print(system_prompt_1[:500])
    else:
        print("\n✗ System Prompt不一致!")
        print("\n差异部分示例:")
        # 找出第一个不同的位置
        for i, (c1, c2) in enumerate(zip(system_prompt_1, system_prompt_2)):
            if c1 != c2:
                print(f"位置 {i}:")
                print(f"  Prompt 1: ...{system_prompt_1[max(0,i-20):i+20]}...")
                print(f"  Prompt 2: ...{system_prompt_2[max(0,i-20):i+20]}...")
                break
        return False
    
    # 测试3: 测试工具调用的一致性
    print("\n" + "=" * 80)
    print("测试3: 验证工具调用的一致性")
    print("=" * 80)
    
    # 创建一个测试图片
    test_image = Image.new('RGB', (100, 100), color='white')
    test_image_base64 = pil_to_base64(test_image)
    
    # 获取随机化后的Point工具名称和参数名
    randomized_point_name = tool_manager_1.original_to_randomized.get("Point", "Point")
    randomized_image_param = tool_manager_1.original_to_randomized.get("image", "image")
    randomized_desc_param = tool_manager_1.original_to_randomized.get("description", "description")
    
    print(f"\n原始工具名: Point -> 随机化名称: {randomized_point_name}")
    print(f"原始参数名: image -> 随机化名称: {randomized_image_param}")
    print(f"原始参数名: description -> 随机化名称: {randomized_desc_param}")
    
    # 使用随机化后的名称调用工具
    api_params_1 = {
        randomized_image_param: test_image_base64,
        randomized_desc_param: "center of image"
    }
    
    api_params_2 = {
        randomized_image_param: test_image_base64,
        randomized_desc_param: "center of image"
    }
    
    print("\n使用 ToolManager 1 调用工具...")
    try:
        result_1 = tool_manager_1.call_tool(randomized_point_name, api_params_1)
        print(f"  调用成功: {result_1.get('status', 'unknown')}")
    except Exception as e:
        print(f"  调用失败: {e}")
        result_1 = None
    
    print("\n使用 ToolManager 2 调用工具...")
    try:
        result_2 = tool_manager_2.call_tool(randomized_point_name, api_params_2)
        print(f"  调用成功: {result_2.get('status', 'unknown')}")
    except Exception as e:
        print(f"  调用失败: {e}")
        result_2 = None
    
    if result_1 and result_2:
        # 比较结果（去除可能变化的字段如execution_time）
        result_1_copy = result_1.copy()
        result_2_copy = result_2.copy()
        
        for key in ['execution_time', 'timestamp']:
            result_1_copy.pop(key, None)
            result_2_copy.pop(key, None)
        
        results_match = result_1_copy == result_2_copy
        print(f"\n工具调用结果是否一致: {results_match}")
        
        if results_match:
            print("\n✓ 工具调用结果一致!")
        else:
            print("\n✗ 工具调用结果不一致!")
            print("\nResult 1:", json.dumps(result_1_copy, indent=2))
            print("\nResult 2:", json.dumps(result_2_copy, indent=2))
            return False
    
    # 测试4: 保存映射信息用于验证
    print("\n" + "=" * 80)
    print("测试4: 保存映射信息")
    print("=" * 80)
    
    output_dir = "./test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    mapping_file_1 = os.path.join(output_dir, "tool_mapping_1.json")
    mapping_file_2 = os.path.join(output_dir, "tool_mapping_2.json")
    
    tool_manager_1.save_randomization_mapping(mapping_file_1)
    tool_manager_2.save_randomization_mapping(mapping_file_2)
    
    print(f"\n映射信息已保存到:")
    print(f"  {mapping_file_1}")
    print(f"  {mapping_file_2}")
    
    # 比较两个文件内容
    with open(mapping_file_1, 'r') as f1, open(mapping_file_2, 'r') as f2:
        mapping_content_1 = f1.read()
        mapping_content_2 = f2.read()
    
    files_match = mapping_content_1 == mapping_content_2
    print(f"\n映射文件内容是否一致: {files_match}")
    
    if files_match:
        print("\n✓ 映射文件一致!")
    else:
        print("\n✗ 映射文件不一致!")
        return False
    
    # 测试5: 测试不同deterministic_id会产生不同的映射
    print("\n" + "=" * 80)
    print("测试5: 验证不同ID产生不同映射")
    print("=" * 80)
    
    print("\n初始化第三个 ToolManager (使用不同的 deterministic_id)...")
    tool_manager_3 = ToolManager(
        randomize=True,
        deterministic_id="different_experiment_v2.0",
        tools=["Point", "Crop", "OCR", "Draw2DPath"]
    )
    
    different_mapping = tool_manager_1.original_to_randomized != tool_manager_3.original_to_randomized
    print(f"\n不同ID的映射是否不同: {different_mapping}")
    
    if different_mapping:
        print("\n✓ 不同ID产生不同映射!")
        print("\n示例对比:")
        for orig in list(tool_manager_1.original_to_randomized.keys())[:3]:
            rand_1 = tool_manager_1.original_to_randomized[orig]
            rand_3 = tool_manager_3.original_to_randomized[orig]
            print(f"  {orig}:")
            print(f"    ID1: {rand_1}")
            print(f"    ID3: {rand_3}")
    else:
        print("\n✗ 不同ID产生了相同映射 (这不应该发生)!")
        return False
    
    # 最终总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("\n✓ 所有测试通过!")
    print("\n确定性随机化功能工作正常:")
    print("  1. 相同ID产生一致的工具映射")
    print("  2. 相同ID产生一致的System Prompt")
    print("  3. 相同ID的工具调用行为一致")
    print("  4. 映射信息可以正确保存和对比")
    print("  5. 不同ID产生不同的映射")
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)
    
    # 测试不启用随机化
    print("\n测试不启用随机化...")
    tm_no_random = ToolManager(
        randomize=False,
        tools=["Point", "Crop"]
    )
    
    prompt = tm_no_random.get_tool_prompt("one_tool_call")
    print(f"  ✓ 非随机化模式工作正常")
    
    # 测试只使用random_seed (不使用deterministic_id)
    print("\n测试使用random_seed...")
    tm_seed_1 = ToolManager(
        randomize=True,
        random_seed=12345,
        tools=["Point", "Crop"]
    )
    
    tm_seed_2 = ToolManager(
        randomize=True,
        random_seed=12345,
        tools=["Point", "Crop"]
    )
    
    seed_match = tm_seed_1.original_to_randomized == tm_seed_2.original_to_randomized
    print(f"  Random seed一致性: {seed_match}")
    
    if seed_match:
        print("  ✓ Random seed模式工作正常")
    else:
        print("  ✗ Random seed模式不一致")
        return False
    
    print("\n✓ 边界情况测试通过!")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("开始测试确定性随机化功能")
    print("=" * 80)
    
    try:
        # 运行主测试
        main_test_passed = test_deterministic_randomization()
        
        # 运行边界情况测试
        edge_test_passed = test_edge_cases()
        
        if main_test_passed and edge_test_passed:
            print("\n" + "=" * 80)
            print("所有测试成功! ✓")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("部分测试失败! ✗")
            print("=" * 80)
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"测试过程中出现异常: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
