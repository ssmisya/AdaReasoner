import cv2
import supervision as sv
import numpy as np

def draw_grounding_dino_box(image_path_or_array, gd_output_dict):
    """
    Draws bounding boxes from Grounding DINO output onto an image.

    Args:
        image_path_or_array (str or np.ndarray): Path to the input image or an image loaded as a NumPy array.
        gd_output_dict (dict): The Grounding DINO output dictionary.
                                Example: {'text': {'boxes': [[0.07, 0.23, 0.95, 0.71]],
                                                    'logits': [0.87],
                                                    'phrases': ['car'],
                                                    'size': [533, 800]}, 'error_code': 0}

    Returns:
        np.ndarray: The image with bounding boxes drawn.
    """
    # 1. 加载图像
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        # 如果传入的是NumPy数组，则直接使用，但最好复制一份以免修改原始数组
        image = image_path_or_array.copy()
    else:
        raise TypeError("image_path_or_array must be a string path or a numpy array.")

    # 2. 从 Grounding DINO 输出中提取数据
    text_data = gd_output_dict.get('text')
    if not text_data:
        print("Warning: 'text' key not found in Grounding DINO output or it's empty. No boxes to draw.")
        return image # 返回原始图像

    boxes_normalized = np.array(text_data['boxes']) # 归一化的xyxy坐标
    logits = np.array(text_data['logits'])         # 置信度
    phrases = text_data['phrases']                  # 文本短语
    
    # Grounding DINO 的 'size' 字段通常是 [height, width]
    original_image_height, original_image_width = text_data['size']

    # 3. 将归一化的坐标转换为像素坐标
    # supervision 的 Detections 期望像素坐标
    boxes_pixel = boxes_normalized * np.array([original_image_width, original_image_height, original_image_width, original_image_height])
    boxes_pixel = boxes_pixel.astype(int) # 转换为整数，因为像素是整数

    # 4. 为 supervision 的 Detections 对象准备数据
    #    由于 Grounding DINO 返回的是短语而非类别ID，我们需要将每个短语视为一个“类别”，
    #    并生成一个唯一的class_id，以便 BoxAnnotator 可以为不同短语分配不同颜色。
    
    # 获取所有唯一的短语，并进行排序以保持 class_id 的一致性
    unique_phrases = sorted(list(set(phrases)))
    
    # 创建一个从短语到整数ID的映射
    phrase_to_class_id = {phrase: i for i, phrase in enumerate(unique_phrases)}
    
    # 为每个检测结果找到对应的 class_id
    class_ids = np.array([phrase_to_class_id[p] for p in phrases])

    # 创建 supervision 的 Detections 对象
    detections = sv.Detections(
        xyxy=boxes_pixel,
        confidence=logits,
        class_id=class_ids # 传入生成的 class_id
    )

    # 5. 初始化 BoxAnnotator
    #    传入 class_names 参数，这样 BoxAnnotator 就可以根据 class_id 自动显示短语和分配颜色。
    box_annotator = sv.BoxAnnotator(
        thickness=2,          # 边界框线条粗细
        text_thickness=1,     # 文本线条粗细
        text_padding=4,       # 文本填充
        text_scale=0.7,       # 文本大小
        class_names=unique_phrases # 传入唯一的短语列表作为类别名称
    )

    # 6. 准备自定义标签文本 (例如："car 0.87")
    #    这将覆盖 BoxAnnotator 默认的只显示类别名称的行为，让你可以显示置信度
    labels = [
        f"{phrase} {confidence:.2f}"
        for phrase, confidence in zip(phrases, logits)
    ]

    # 7. 在图像上绘制边界框和标签
    annotated_image = box_annotator.annotate(
        scene=image.copy(),  # 传入图像的副本，避免修改原始图像
        detections=detections,
        labels=labels        # 传入自定义的标签列表
    )

    return annotated_image

# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 创建一个虚拟图片用于测试 (如果你的图片不存在)
    # 假设你的图片是 800 像素高，533 像素宽
    dummy_image_height = 800
    dummy_image_width = 533
    dummy_image = np.zeros((dummy_image_height, dummy_image_width, 3), dtype=np.uint8) # 全黑图片
    
    # 在图片中心画一个白色的圆，让它看起来不是完全空的
    center_x, center_y = dummy_image_width // 2, dummy_image_height // 2
    cv2.circle(dummy_image, (center_x, center_y), 50, (255, 255, 255), -1)

    dummy_image_path = "my_image.jpg"
    cv2.imwrite(dummy_image_path, dummy_image)
    print(f"Created a dummy image at: {dummy_image_path}")

    # 2. 你的 Grounding DINO 输出数据
    grounding_dino_output_data = {
        'text': {
            'boxes': [[0.07, 0.23, 0.95, 0.71]],
            'logits': [0.87],
            'phrases': ['car'],
            'size': [dummy_image_height, dummy_image_width] # 注意这里的顺序：[height, width]
        },
        'error_code': 0
    }

    # 3. 调用函数进行标注
    print("\nAnnotating image...")
    # 方式一：传入图片路径
    annotated_image_from_path = draw_grounding_dino_box(dummy_image_path, grounding_dino_output_data)

    # 方式二：传入已加载的图片 NumPy 数组 (例如：如果你已经用 cv2.imread 加载了图片)
    # loaded_image = cv2.imread(dummy_image_path)
    # annotated_image_from_array = draw_grounding_dino_box(loaded_image, grounding_dino_output_data)

    print("Annotation complete. Displaying image...")

    # 4. 显示标注后的图片
    cv2.imshow("Annotated Image", annotated_image_from_path)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows() # 关闭所有OpenCV窗口

    # 5. 保存标注后的图片
    output_image_path = "annotated_result.jpg"
    cv2.imwrite(output_image_path, annotated_image_from_path)
    print(f"Annotated image saved to: {output_image_path}")