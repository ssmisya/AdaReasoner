# balance_test.py
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} records")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def analyze_path_balance(data: List[Dict[str, Any]]) -> Dict:
    """Analyze safety ratios for different path lengths"""
    # Statistics by size and length
    stats_by_size_length = defaultdict(lambda: defaultdict(lambda: {"safe": 0, "unsafe": 0}))
    
    # Statistics by length (all sizes)
    stats_by_length = defaultdict(lambda: {"safe": 0, "unsafe": 0})
    
    # Overall statistics
    total_stats = {"safe": 0, "unsafe": 0}
    
    # List for storing length and safety information for further analysis
    path_data = []
    
    for item in data:
        if "path_drawings" in item and "random" in item["path_drawings"]:
            random_path = item["path_drawings"]["random"]
            
            if "path" in random_path and "is_safe" in random_path:
                path = random_path["path"]
                is_safe = random_path["is_safe"]
                size = item["size"]
                
                # Calculate path length
                path_length = len(path.split(','))
                
                # Collect data
                path_data.append({
                    "size": size,
                    "length": path_length,
                    "is_safe": is_safe
                })
                
                # Update statistics
                if is_safe:
                    stats_by_size_length[size][path_length]["safe"] += 1
                    stats_by_length[path_length]["safe"] += 1
                    total_stats["safe"] += 1
                else:
                    stats_by_size_length[size][path_length]["unsafe"] += 1
                    stats_by_length[path_length]["unsafe"] += 1
                    total_stats["unsafe"] += 1
    
    # Calculate percentages
    length_percentages = {}
    for length, stats in stats_by_length.items():
        total = stats["safe"] + stats["unsafe"]
        if total > 0:
            safe_percent = (stats["safe"] / total) * 100
            unsafe_percent = (stats["unsafe"] / total) * 100
            length_percentages[length] = {
                "safe_percent": safe_percent,
                "unsafe_percent": unsafe_percent,
                "total": total
            }
    
    # Calculate percentages for each size and length
    size_length_percentages = {}
    for size, lengths in stats_by_size_length.items():
        size_length_percentages[size] = {}
        for length, stats in lengths.items():
            total = stats["safe"] + stats["unsafe"]
            if total > 0:
                safe_percent = (stats["safe"] / total) * 100
                unsafe_percent = (stats["unsafe"] / total) * 100
                size_length_percentages[size][length] = {
                    "safe_percent": safe_percent,
                    "unsafe_percent": unsafe_percent,
                    "total": total
                }
    
    # Overall percentages
    total = total_stats["safe"] + total_stats["unsafe"]
    if total > 0:
        total_safe_percent = (total_stats["safe"] / total) * 100
        total_unsafe_percent = (total_stats["unsafe"] / total) * 100
    else:
        total_safe_percent = 0
        total_unsafe_percent = 0
    
    return {
        "by_length": length_percentages,
        "by_size_length": size_length_percentages,
        "overall": {
            "safe_percent": total_safe_percent,
            "unsafe_percent": total_unsafe_percent,
            "total": total
        },
        "path_data": path_data
    }

def plot_balance_charts(stats: Dict, output_prefix: str = "balance"):
    """Plot balance charts"""
    # 1. Plot safe/unsafe ratio by length
    lengths = sorted(stats["by_length"].keys())
    safe_percentages = [stats["by_length"][length]["safe_percent"] for length in lengths]
    unsafe_percentages = [stats["by_length"][length]["unsafe_percent"] for length in lengths]
    counts = [stats["by_length"][length]["total"] for length in lengths]
    
    plt.figure(figsize=(12, 8))
    width = 0.35
    x = np.arange(len(lengths))
    
    bars1 = plt.bar(x - width/2, safe_percentages, width, label='Safe Paths')
    bars2 = plt.bar(x + width/2, unsafe_percentages, width, label='Unsafe Paths')
    
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Path Length')
    plt.ylabel('Percentage (%)')
    plt.title('Safe/Unsafe Ratio for Different Path Lengths')
    plt.xticks(x, lengths)
    plt.legend()
    
    # Display sample count on each bar
    for i, v in enumerate(counts):
        plt.text(i - width/2, 5, f"n={v}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_by_length.png")
    
    # 2. Plot heatmap by size and length
    sizes = sorted(stats["by_size_length"].keys())
    
    # Find all lengths
    all_lengths = set()
    for size in sizes:
        all_lengths.update(stats["by_size_length"][size].keys())
    all_lengths = sorted(all_lengths)
    
    # Create heatmap data
    heatmap_data = []
    annotations = []
    
    for size in sizes:
        row_data = []
        row_annot = []
        for length in all_lengths:
            if length in stats["by_size_length"][size]:
                row_data.append(stats["by_size_length"][size][length]["safe_percent"])
                row_annot.append(f"{stats['by_size_length'][size][length]['total']}")
            else:
                row_data.append(np.nan)  # No data
                row_annot.append("")
        heatmap_data.append(row_data)
        annotations.append(row_annot)
    
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
    
    # Use safe path percentage as heatmap value
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set x and y axis labels
    ax.set_xticks(np.arange(len(all_lengths)))
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_xticklabels(all_lengths)
    ax.set_yticklabels(sizes)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations
    for i in range(len(sizes)):
        for j in range(len(all_lengths)):
            if j < len(annotations[i]) and annotations[i][j]:
                ax.text(j, i, f"{heatmap_data[i][j]:.1f}%\nn={annotations[i][j]}", 
                        ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar and title
    plt.colorbar(im, label="Safe Path Percentage")
    plt.xlabel("Path Length")
    plt.ylabel("Map Size")
    plt.title("Safe Path Percentage by Size and Length")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png")
    
    # 3. Length distribution plot
    lengths = [item["length"] for item in stats["path_data"]]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(1, max(lengths)+2), alpha=0.7, edgecolor='black')
    plt.xlabel('Path Length')
    plt.ylabel('Sample Count')
    plt.title('Path Length Distribution')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(range(1, max(lengths)+1))
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_length_distribution.png")
    
    # 4. Scatter plot by length and safety
    plt.figure(figsize=(12, 6))
    
    safe_paths = [(item["length"], item["size"]) for item in stats["path_data"] if item["is_safe"]]
    unsafe_paths = [(item["length"], item["size"]) for item in stats["path_data"] if not item["is_safe"]]
    
    # Add small random jitter to avoid overlapping
    jitter = 0.2
    
    if safe_paths:
        x_safe, y_safe = zip(*safe_paths)
        x_safe = [x + (np.random.rand() - 0.5) * jitter for x in x_safe]
        plt.scatter(x_safe, y_safe, alpha=0.6, label='Safe Paths', color='green')
    
    if unsafe_paths:
        x_unsafe, y_unsafe = zip(*unsafe_paths)
        x_unsafe = [x + (np.random.rand() - 0.5) * jitter for x in x_unsafe]
        plt.scatter(x_unsafe, y_unsafe, alpha=0.6, label='Unsafe Paths', color='red')
    
    plt.xlabel('Path Length')
    plt.ylabel('Map Size')
    plt.title('Relationship between Path Length, Map Size and Safety')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max([item["length"] for item in stats["path_data"]])+1))
    plt.yticks(range(min([item["size"] for item in stats["path_data"]]), max([item["size"] for item in stats["path_data"]])+1))
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_scatter.png")
    
    plt.close('all')

def print_statistics(stats: Dict):
    """Print statistics"""
    print("\n===== Overall Statistics =====")
    print(f"Total samples: {stats['overall']['total']}")
    print(f"Safe paths: {stats['overall']['safe_percent']:.2f}%")
    print(f"Unsafe paths: {stats['overall']['unsafe_percent']:.2f}%")
    
    print("\n===== Statistics by Path Length =====")
    for length in sorted(stats["by_length"].keys()):
        data = stats["by_length"][length]
        print(f"Length {length}: Total={data['total']}, Safe={data['safe_percent']:.2f}%, Unsafe={data['unsafe_percent']:.2f}%")
    
    print("\n===== Statistics by Map Size =====")
    for size in sorted(stats["by_size_length"].keys()):
        print(f"\nMap size {size}x{size}:")
        for length in sorted(stats["by_size_length"][size].keys()):
            data = stats["by_size_length"][size][length]
            print(f"  Length {length}: Total={data['total']}, Safe={data['safe_percent']:.2f}%, Unsafe={data['unsafe_percent']:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Verify balance between path length and safety')
    parser.add_argument('--data', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v1/dataset.jsonl', 
                        help='Dataset file path')
    parser.add_argument('--output', type=str, default='balance', 
                        help='Output chart file prefix')
    
    args = parser.parse_args()
    
    # Load data
    data = load_jsonl(args.data)
    if not data:
        return
    
    # Analyze balance
    stats = analyze_path_balance(data)
    
    # Print statistics
    print_statistics(stats)
    
    # Plot charts
    plot_balance_charts(stats, args.output)
    
    print(f"Analysis complete, charts generated: {args.output}_*.png")

if __name__ == "__main__":
    main()