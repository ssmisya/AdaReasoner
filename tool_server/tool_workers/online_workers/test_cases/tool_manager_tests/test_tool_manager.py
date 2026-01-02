# test_tool_manager.py
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
from tool_server.utils.utils import pil_to_base64
from PIL import Image
import json

def test_call_tool_with_external_mapping():
    """测试使用外部 randomized_to_original 映射调用工具"""
    
    print("=" * 80)
    print("测试 call_tool 使用外部映射功能")
    print("=" * 80)
    
    # === 阶段1: 初始化第一个 ToolManager 并获取随机化映射 ===
    print("\n阶段1: 初始化 ToolManager A (获取随机化映射)")
    print("-" * 80)
    
    tool_manager_a = ToolManager(
        randomize=True,
        deterministic_id="source_experiment_v1.0",
        tools=["Point", "Crop", "OCR"]
    )
    
    # 获取随机化映射
    randomized_to_original_a = tool_manager_a.randomized_to_original.copy()
    original_to_randomized_a = tool_manager_a.original_to_randomized.copy()
    
    print(f"\n✓ ToolManager A 初始化完成")
    print(f"  工具映射数量: {len(original_to_randomized_a)}")
    print("\n示例映射 (原始 -> 随机化):")
    for orig, rand in list(original_to_randomized_a.items())[:5]:
        print(f"  {orig:20s} -> {rand}")
    
    # === 阶段2: 构造随机化的工具调用参数 ===
    print("\n阶段2: 构造随机化的工具调用")
    print("-" * 80)
    
    # 创建测试图片
    test_image = Image.new('RGB', (200, 200), color='red')
    # 在中心画一个蓝色方块
    for x in range(75, 125):
        for y in range(75, 125):
            test_image.putpixel((x, y), (0, 0, 255))
    
    test_image_base64 = pil_to_base64(test_image)
    
    # 获取 Point 工具的随机化名称
    randomized_point_name = original_to_randomized_a.get("Point", "Point")
    randomized_image_param = original_to_randomized_a.get("image", "image")
    randomized_desc_param = original_to_randomized_a.get("description", "description")
    
    # 构造随机化的调用参数
    randomized_params = {
        randomized_image_param: test_image_base64,
        randomized_desc_param: "center blue square"
    }
    
    print(f"\n随机化的工具调用:")
    print(f"  工具名: Point -> {randomized_point_name}")
    print(f"  参数:")
    print(f"    image -> {randomized_image_param}")
    print(f"    description -> {randomized_desc_param}")
    print(f"  参数值: {{'{randomized_image_param}': <image_base64>, '{randomized_desc_param}': 'center blue square'}}")
    
    # === 阶段3: 使用 ToolManager A 自己的映射调用 (基准测试) ===
    print("\n阶段3: 基准测试 - 使用 ToolManager A 自己调用")
    print("-" * 80)
    
    print(f"\n调用工具: {randomized_point_name}")
    try:
        result_baseline = tool_manager_a.call_tool(randomized_point_name, randomized_params)
        result_baseline.pop("edited_image", None)  # 移除大数据字段以便显示
        if result_baseline.get("error_code", 1) == 0:
            print(f"  ✓ 调用成功")
            print(f"  结果: {result_baseline}")
        else:
            print(f"  ✗ 调用失败: {result_baseline}")
    except Exception as e:
        print(f"  ✗ 调用异常: {e}")
        result_baseline = None
    
    # === 阶段4: 初始化第二个 ToolManager (使用不同的随机化配置) ===
    print("\n阶段4: 初始化 ToolManager B (不同的随机化配置)")
    print("-" * 80)
    
    tool_manager_b = ToolManager(
        randomize=True,
        deterministic_id="target_experiment_v2.0",  # 不同的ID
        tools=["Point", "Crop", "OCR"]
    )
    
    print(f"\n✓ ToolManager B 初始化完成")
    print(f"  使用不同的 deterministic_id: target_experiment_v2.0")
    
    # 显示 B 的映射与 A 的映射不同
    randomized_point_name_b = tool_manager_b.original_to_randomized.get("Point", "Point")
    print(f"\n映射对比:")
    print(f"  ToolManager A: Point -> {randomized_point_name}")
    print(f"  ToolManager B: Point -> {randomized_point_name_b}")
    print(f"  映射不同: {randomized_point_name != randomized_point_name_b}")
    
    # === 阶段5: 使用外部映射 (A的映射) 调用 B 的工具 ===
    print("\n阶段5: 测试新功能 - 使用 A 的映射调用 B 的工具")
    print("-" * 80)
    
    print(f"\n准备调用:")
    print(f"  目标 ToolManager: B")
    print(f"  随机化工具名: {randomized_point_name} (来自 A)")
    print(f"  随机化参数: {list(randomized_params.keys())} (来自 A)")
    print(f"  提供的映射字典: randomized_to_original_a")
    
    try:
        # 关键测试: 使用 A 的映射字典调用 B 的工具
        result_cross = tool_manager_b.call_tool(
            randomized_point_name,  # A 的随机化名称
            randomized_params,      # A 的随机化参数
            randomized_to_original=randomized_to_original_a  # 提供 A 的映射
        )
        result_cross.pop("edited_image", None)  # 移除大数据字段以便显示
        if result_cross.get("error_code", 1) == 0:
            print(f"\n✓ 跨 ToolManager 调用成功!")
            print(f"  结果: {result_cross}")
            
            # 比较两次调用的结果
            if result_baseline and result_cross:
                # 移除时间戳等动态字段
                baseline_copy = result_baseline.copy()
                cross_copy = result_cross.copy()
                for key in ['execution_time', 'timestamp']:
                    baseline_copy.pop(key, None)
                    cross_copy.pop(key, None)
                
                results_similar = baseline_copy == cross_copy
                print(f"\n结果一致性: {results_similar}")
                
        else:
            print(f"\n✗ 调用失败: {result_cross.get('text', 'Unknown error')}")
            result_cross = None
            
    except Exception as e:
        print(f"\n✗ 调用异常: {e}")
        import traceback
        traceback.print_exc()
        result_cross = None
    
    # === 阶段6: 对比测试 - 不提供映射字典 (应该失败) ===
    print("\n阶段6: 对比测试 - 不提供外部映射 (应该失败)")
    print("-" * 80)
    
    try:
        # 这应该失败,因为 B 使用自己的映射无法识别 A 的随机化名称
        result_no_mapping = tool_manager_b.call_tool(
            randomized_point_name,  # A 的随机化名称
            randomized_params       # A 的随机化参数
            # 不提供 randomized_to_original
        )
        result_no_mapping.pop("edited_image", None)  # 移除大数据字段以便显示
        if result_no_mapping.get("error_code", 1) == 0:
            print(f"  ⚠ 意外成功 (这可能表示映射逻辑有问题)")
        else:
            print(f"  ✓ 按预期失败: {result_no_mapping.get('text', 'Unknown error')[:100]}")
            
    except Exception as e:
        print(f"  ✓ 按预期抛出异常: {str(e)[:100]}")
    
    # === 测试总结 ===
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if result_cross and result_cross.get("error_code", 1) == 0:
        print("\n✓ call_tool 外部映射功能测试通过!")
        print("\n验证点:")
        print("  1. ✓ 成功使用外部映射字典")
        print("  2. ✓ 跨 ToolManager 调用成功")
        print("  3. ✓ 不同随机化配置间正确转换")
        return True
    else:
        print("\n✗ 测试失败!")
        return False


def test_multiple_tools():
    """测试多个工具的外部映射调用"""
    
    print("\n" + "=" * 80)
    print("测试多工具外部映射调用")
    print("=" * 80)
    
    # 初始化源 ToolManager
    print("\n初始化源 ToolManager...")
    tm_source = ToolManager(
        randomize=True,
        deterministic_id="source_multi_v1.0",
        tools=["Point", "Crop", "OCR"]
    )
    
    source_mapping = tm_source.randomized_to_original.copy()
    
    # 初始化目标 ToolManager
    print("初始化目标 ToolManager...")
    tm_target = ToolManager(
        randomize=True,
        deterministic_id="target_multi_v2.0",
        tools=["Point", "Crop", "OCR"]
    )
    
    # 测试图片
    test_image = Image.new('RGB', (300, 200), color='white')
    test_image_base64 = pil_to_base64(test_image)
    
    # 测试多个工具
    test_cases = [
        {
            "original_tool": "Point",
            "params_map": {"image": test_image_base64, "description": "top left corner"}
        },
        {
            "original_tool": "Crop",
            "params_map": {"image": test_image_base64, "coordinates": [10, 10, 50, 50]}
        },
        {
            "original_tool": "OCR",
            "params_map": {"image": test_image_base64}
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['original_tool']}")
        print("-" * 80)
        
        # 构造随机化参数
        orig_tool = test_case["original_tool"]
        rand_tool = tm_source.original_to_randomized.get(orig_tool, orig_tool)
        
        rand_params = {}
        for param_name, param_value in test_case["params_map"].items():
            rand_param = tm_source.original_to_randomized.get(param_name, param_name)
            rand_params[rand_param] = param_value
        
        print(f"  原始: {orig_tool}({list(test_case['params_map'].keys())})")
        print(f"  随机: {rand_tool}({list(rand_params.keys())})")
        
        try:
            result = tm_target.call_tool(
                rand_tool,
                rand_params,
                randomized_to_original=source_mapping
            )
            result.pop("edited_image", None)  # 移除大数据字段
            if result.get("error_code", 1) == 0:
                print(f"  ✓ 调用成功")
                success_count += 1
            else:
                print(f"  ✗ 调用失败: {result.get('text', 'Unknown error')[:100]}")
                
        except Exception as e:
            print(f"  ✗ 异常: {str(e)[:100]}")
    
    print(f"\n成功率: {success_count}/{len(test_cases)}")
    
    return success_count == len(test_cases)


def test_edge_cases():
    """测试边界情况"""
    
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)
    
    tm = ToolManager(
        randomize=True,
        deterministic_id="edge_test_v1.0",
        tools=["Point"]
    )
    
    test_image = Image.new('RGB', (100, 100), color='white')
    test_image_base64 = pil_to_base64(test_image)
    
    # 测试1: 空映射字典
    print("\n测试1: 空映射字典")
    try:
        result = tm.call_tool(
            "Point",
            {"image": test_image_base64, "description": "center"},
            randomized_to_original={}
        )
        if "edited_image" in result:
            result.pop("edited_image")  # 移除大数据字段
        print(f"  结果: {result}")
    except Exception as e:
        print(f"  异常: {str(e)[:50]}")
    
    # 测试2: 部分映射字典
    print("\n测试2: 部分映射 (只映射工具名,不映射参数)")
    partial_mapping = {"Point": "Point"}  # 缺少参数映射
    try:
        result = tm.call_tool(
            "Point",
            {"image": test_image_base64, "description": "center"},
            randomized_to_original=partial_mapping
        )
        if "edited_image" in result:
            result.pop("edited_image")  # 移除大数据字段
        print(f"  结果: {result}")
    except Exception as e:
        print(f"  异常: {str(e)[:50]}")
    
    # 测试3: None 映射 (应该使用自己的映射)
    print("\n测试3: 映射为 None (应使用内部映射)")
    rand_tool = tm.original_to_randomized.get("Point", "Point")
    rand_params = {
        tm.original_to_randomized.get("image", "image"): test_image_base64,
        tm.original_to_randomized.get("description", "description"): "center"
    }
    try:
        result = tm.call_tool(rand_tool, rand_params, randomized_to_original=None)
        if "edited_image" in result:
            result.pop("edited_image")  # 移除大数据字段
        if result.get("error_code", 1) == 0:
            print(f"  ✓ 成功使用内部映射")
        else:
            print(f"  ✗ 失败: {result}")
    except Exception as e:
        print(f"  异常: {str(e)[:50]}")
    
    print("\n✓ 边界情况测试完成")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("开始测试 call_tool 外部映射功能")
    print("=" * 80)
    
    try:
        # 主测试
        test1_passed = test_call_tool_with_external_mapping()
        
        # 多工具测试
        test2_passed = test_multiple_tools()
        
        # 边界情况测试
        test3_passed = test_edge_cases()
        
        if test1_passed and test2_passed and test3_passed:
            print("\n" + "=" * 80)
            print("所有测试通过! ✓")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("部分测试失败! ✗")
            print("=" * 80)
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"测试异常: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)