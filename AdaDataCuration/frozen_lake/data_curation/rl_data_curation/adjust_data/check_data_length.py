import pandas as pd
import argparse
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class DataLengthChecker:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        """初始化检查器"""
        print(f"加载模型和处理器: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = None  # 不需要加载完整模型，只用processor
        


    def analyze_parquet(self, parquet_file, output_dir="./analysis_results"):
        """分析parquet文件"""
        print(f"\n{'='*60}")
        print(f"开始分析文件: {parquet_file}")
        print(f"{'='*60}\n")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("读取数据...")
        
        try:
            # 使用 pandas 读取
            df = pd.read_parquet(parquet_file)
            print(f"数据集大小: {len(df)} 条记录\n")
            
            prompt_lengths = []
            token_lengths = []
            max_token_item = None
            max_token_length = 0
            
            print("开始处理数据...")
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="分析进度"):
                try:
                    item = row.to_dict()
                    
                    # 确保prompt字段被正确转换
                    if 'prompt' in item and isinstance(item['prompt'], np.ndarray):
                        item['prompt'] = item['prompt'].tolist()
                    
                    # 确保images字段被正确转换
                    if 'images' in item and isinstance(item['images'], np.ndarray):
                        item['images'] = item['images'].tolist()
                    
                    test_json = json.loads(item["randomized_to_original"])
                    
                    # 计算长度
                    prompt_len = self.calculate_prompt_length(item['prompt'])
                    prompt_lengths.append(prompt_len)
                    
                    token_len, messages, images = self.process_item_to_tokens(item)
                    if token_len is not None:
                        token_lengths.append(token_len)
                        
                        if token_len > max_token_length:
                            max_token_length = token_len
                            max_token_item = {
                                'index': idx,
                                'token_length': token_len,
                                'prompt_length': prompt_len,
                                'messages': messages,
                                'images': images,
                                'item': item
                            }
                    
                    if (idx + 1) % 1000 == 0:
                        print(f"已处理 {idx + 1}/{len(df)} 条数据")
                        
                except Exception as e:
                    print(f"处理第 {idx} 条数据时出错: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            raise
        
        # 后续处理...
        if len(token_lengths) == 0:
            print("❌ 没有成功处理任何数据")
            return None, None
        
        results = self.generate_statistics(prompt_lengths, token_lengths)
        self.print_statistics(results)
        self.save_results(results, output_dir)
        self.visualize_results(prompt_lengths, token_lengths, output_dir)
        
        if max_token_item:
            self.save_longest_example(max_token_item, output_dir)
        
        return results, max_token_item    
    
    
    
    def calculate_prompt_length(self, prompt):
        """计算prompt文本长度"""
        # 处理不同类型的prompt
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        elif isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except:
                try:
                    import ast
                    prompt = ast.literal_eval(prompt)
                except:
                    return 0
        
        # 确保是列表类型
        if not isinstance(prompt, list):
            return 0
        
        total_length = 0
        for message in prompt:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if isinstance(content, str):
                    total_length += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            total_length += len(item.get('text', ''))
        return total_length

    def process_item_to_tokens(self, item):
        """处理单个数据项，返回token数量"""
        try:
            # 准备messages - 统一处理各种类型
            prompt = item['prompt']
            
            # 类型转换：numpy.ndarray -> list
            if isinstance(prompt, np.ndarray):
                messages = prompt.tolist()
            elif isinstance(prompt, str):
                # 字符串类型尝试JSON解析
                try:
                    messages = json.loads(prompt)
                except json.JSONDecodeError:
                    try:
                        import ast
                        messages = ast.literal_eval(prompt)
                    except:
                        print(f"无法解析prompt字符串")
                        return None, None, None
            elif isinstance(prompt, list):
                messages = prompt
            else:
                print(f"未知的prompt类型: {type(prompt)}")
                return None, None, None
            
            # 确保messages是有效的列表
            if not isinstance(messages, list) or len(messages) == 0:
                print(f"messages格式无效")
                return None, None, None
            
            # 处理图片
            images_data = []
            if 'images' in item and item['images'] is not None:
                images = item['images']
                
                # 处理numpy array类型的images
                if isinstance(images, np.ndarray):
                    images = images.tolist()
                
                # 确保images是列表
                if not isinstance(images, (list, tuple)):
                    images = [images]
                
                for img_dict in images:
                    try:
                        if isinstance(img_dict, dict) and 'bytes' in img_dict:
                            img_bytes = img_dict['bytes']
                            img = Image.open(io.BytesIO(img_bytes))
                            images_data.append(img)
                    except Exception as e:
                        print(f"处理图片时出错: {str(e)}")
                        continue
            
            # 构建processor输入格式
            text_input = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 处理图片信息
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 使用processor处理
            inputs = self.processor(
                text=[text_input],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            
            # 获取token长度
            token_length = inputs['input_ids'].shape[1]
            
            return token_length, messages, images_data
            
        except Exception as e:
            print(f"处理数据项时出错: {str(e)}")
            if isinstance(item, dict):
                print(f"  prompt类型: {type(item.get('prompt'))}")
            return None, None, None

    def generate_statistics(self, prompt_lengths, token_lengths):
        """生成统计信息"""
        # 检查是否有有效数据
        if len(prompt_lengths) == 0:
            print("警告：没有有效的prompt长度数据")
            return None
        
        if len(token_lengths) == 0:
            print("警告：没有有效的token长度数据")
            return None
        
        results = {
            'prompt_length': {
                'min': int(np.min(prompt_lengths)),
                'max': int(np.max(prompt_lengths)),
                'mean': float(np.mean(prompt_lengths)),
                'median': float(np.median(prompt_lengths)),
                'std': float(np.std(prompt_lengths)),
            },
            'token_length': {
                'min': int(np.min(token_lengths)),
                'max': int(np.max(token_lengths)),
                'mean': float(np.mean(token_lengths)),
                'median': float(np.median(token_lengths)),
                'std': float(np.std(token_lengths)),
            },
            'total_samples': len(prompt_lengths),
            'valid_token_samples': len(token_lengths)
        }
        return results
    
    
    def print_statistics(self, results):
        """打印统计结果"""
        print("\n" + "="*60)
        print("统计结果")
        print("="*60)
        
        print("\n📝 Prompt文本长度统计:")
        print(f"  最小值: {results['prompt_length']['min']:,} 字符")
        print(f"  最大值: {results['prompt_length']['max']:,} 字符")
        print(f"  平均值: {results['prompt_length']['mean']:,.2f} 字符")
        print(f"  中位数: {results['prompt_length']['median']:,} 字符")
        print(f"  标准差: {results['prompt_length']['std']:,.2f}")
        
        print("\n🔢 Token长度统计:")
        print(f"  最小值: {results['token_length']['min']:,} tokens")
        print(f"  最大值: {results['token_length']['max']:,} tokens")
        print(f"  平均值: {results['token_length']['mean']:,.2f} tokens")
        print(f"  中位数: {results['token_length']['median']:,} tokens")
        print(f"  标准差: {results['token_length']['std']:,.2f}")
        
        print(f"\n总样本数: {results['total_samples']:,}")
        print("="*60 + "\n")
    
    def save_results(self, results, output_dir):
        """保存统计结果到JSON"""
        output_file = f"{output_dir}/statistics.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 统计结果已保存到: {output_file}")
    
    def visualize_results(self, prompt_lengths, token_lengths, output_dir):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Prompt长度分布直方图
        axes[0, 0].hist(prompt_lengths, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Prompt Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Prompt Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Token长度分布直方图
        axes[0, 1].hist(token_lengths, bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Token Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Token Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prompt长度箱线图
        axes[1, 0].boxplot(prompt_lengths, vert=True)
        axes[1, 0].set_title('Prompt Length Box Plot', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Prompt Length (characters)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Token长度箱线图
        axes[1, 1].boxplot(token_lengths, vert=True)
        axes[1, 1].set_title('Token Length Box Plot', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Token Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = f"{output_dir}/distribution_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存到: {output_file}")
        plt.close()
    
    def save_longest_example(self, max_token_item, output_dir):
        """保存最长的样本"""
        print(f"\n{'='*60}")
        print(f"最长样本信息 (索引: {max_token_item['index']})")
        print(f"{'='*60}")
        print(f"Token长度: {max_token_item['token_length']:,}")
        print(f"Prompt长度: {max_token_item['prompt_length']:,}")
        print(f"图片数量: {len(max_token_item['images'])}")
        
        # 保存详细信息到JSON
        output_data = {
            'index': max_token_item['index'],
            'token_length': max_token_item['token_length'],
            'prompt_length': max_token_item['prompt_length'],
            'num_images': len(max_token_item['images']),
            'messages': max_token_item['messages'],
        }
        
        # 如果有randomized_to_original字段，也保存
        if 'randomized_to_original' in max_token_item['item']:
            try:
                output_data['randomized_to_original'] = json.loads(
                    max_token_item['item']['randomized_to_original']
                )
            except:
                pass
        
        json_file = f"{output_dir}/longest_example.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 最长样本详情已保存到: {json_file}")
        
        # 保存图片
        if max_token_item['images']:
            print(f"\n保存最长样本的图片...")
            for i, img in enumerate(max_token_item['images']):
                img_file = f"{output_dir}/longest_example_image_{i}.png"
                img.save(img_file)
                print(f"  ✓ 图片 {i+1} 已保存到: {img_file}")
        
        # 可视化图片网格
        if max_token_item['images']:
            self.visualize_images(max_token_item['images'], output_dir)
        
        print(f"{'='*60}\n")
    
    def visualize_images(self, images, output_dir):
        """可视化图片网格"""
        n_images = len(images)
        if n_images == 0:
            return
        
        # 计算网格大小
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            ax.imshow(img)
            ax.set_title(f'Image {idx+1}\nSize: {img.size}', fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        output_file = f"{output_dir}/longest_example_images_grid.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 图片网格已保存到: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="检查随机化后的Parquet数据长度统计")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/randomized_rl_data/unified_train_randomized.parquet",
        # default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/unified_train.parquet",
        help="输入Parquet文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./analysis_results",
        help="输出目录"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2.5-VL-7B-Instruct",
        help="Qwen2-VL模型名称"
    )
    
    args = parser.parse_args()
    
    # 创建检查器
    checker = DataLengthChecker(model_name=args.model_name)
    
    # 分析数据
    results, longest_item = checker.analyze_parquet(
        args.input_file,
        args.output_dir
    )
    
    print("\n✅ 分析完成！")
    print(f"所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    # 使用示例:
    # python check_data_length.py \
    #     --input_file /path/to/randomized.parquet \
    #     --output_dir ./analysis_results \
    #     --model_name Qwen/Qwen2-VL-7B-Instruct
    
    main()