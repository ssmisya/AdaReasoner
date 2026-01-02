ACTION_SAFE_PROMPT = """"
You are a maze-solving agent playing a pixelated maze videogame.
Mazes are presented on grid maps, where each tile can be empty land, or contain a player, hole, or goal.
Each of the above tile types are represented as square pixel art images.

In this task, you will analyze a grid-based map and determine if a provided action plan is safe. A safe action plan avoids stepping into holes in the map.
The following figure illustrates the appearances of the player, holes, lands, and the goal within the maze.

<IMAGE-1>

## Moving Rules
- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).
- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.
- Moving off the edge of the map has no effect. The player will remain at the same square.
- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.
- Locating at the grid containing the goal results in victory.
We provide an example to further illustrate the rules.

<IMAGE-2>

In this provided example:
- The player is at Row 1, Column 1;
- The goal is at Row 4, Column 4;
- There are two holes: one at Row 1, Column 2, and another at Row 4, Column 1.
- The player can move DOWN. This is because moving down brings them to Row 2, Column 1, and this cell is safe (without holes).
- Moving UP has no effects. This is because the player is already in the topmost row.
- Similarly, moving LEFT has no effects because the player is already in the left-most column.
- Moving RIGHT places the player at Row 1, Column 2. Since there is a hole at this grid, this move results in a loss.

## Procedure and Output
Your output should include the following parts:
1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.
2. Then, reasoning by following the given action plan. At each step, you should check:
    (a) Where the current move leads the player to (the row and column);
    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?
    (c) Determine if that is a safe action.
3. Output if the action sequence is safe using "<Output> Yes" or "<Output> No". A safe action sequence should not include any unsafe actions.

Now please determine if the action sequence is safe for this given maze:

<TEST-IMAGE>

The action sequence is:

<ACTION-SEQ>
"""


ACTION_SAFE_PROMPT_SHORT = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

<TEST-IMAGE>

The action sequence is:

<ACTION-SEQ>
"""

'''
example_data_jsonl

{"id": "frozenlake_s3_4", "size": 3, "map": ["FHH", "GFF", "FHS"], "path_info": {"gt_path": "ull", "random_path": "ulu", "random_path_safe": false}, "coordinates": {"start": [160.0, 160.0], "holes": [[96.0, 32.0], [160.0, 32.0], [96.0, 160.0]], "goal": [32.0, 96.0]}, "image_paths": {"image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_4.png", "path_image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_4_path.png", "random_path_image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_4_random_path.png"}}


image is the original maze image, which should replace <TEST-IMAGE> in the prompt.

./assets/system-figure-1.png should replace <IMAGE-1> in the prompt.
./assets/system-figure-2.png should replace <IMAGE-2> in the prompt.


came up a prompt that can use path_image as intermediate visual CoT as the solution to the maze.

and the final answer should be "gt_path"
'''

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, Features, Value, Sequence, Image
from huggingface_hub import HfApi
from PIL import Image as PILImage

COT_PROMPT_TEMPLATE = """
Here is the visualization of the path taken by the agent in the maze:

<PATH-IMAGE>
"""
    
def convert_path_to_actions(path_string: str) -> str:
    """Convert path string (e.g., 'ull') to action format (e.g., 'U,L,L')"""
    action_map = {'u': 'U', 'd': 'D', 'l': 'L', 'r': 'R'}
    actions = [action_map[char.lower()] for char in path_string if char.lower() in action_map]
    return ','.join(actions)



def generate_action_plan_dataset(jsonl_file_path: str, output_file_path: str,
                                image1_path: str = "./assets/system-figure-1.png",
                                image2_path: str = "./assets/system-figure-2.png",):
    """
    Generate action plan dataset from maze data using the existing prompt template.
    
    Args:
        jsonl_file_path: Path to the input JSONL file containing maze data
        output_file_path: Path to save the generated action plan dataset
        image1_path: Path to system figure 1
        image2_path: Path to system figure 2
    """
    dataset_entries = []
    
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            for is_gt in [True, False]:
                # Extract information from the maze data
                maze_id = data['id']+ f"_{'gt' if is_gt else 'random'}"
                if is_gt:
                    path = data['path_info']['gt_path']
                    path_image_path = data['image_paths']['path_image']
                else:
                    path = data['path_info']['random_path']
                    path_image_path = data['image_paths']['random_path_image']
                test_image_path = data['image_paths']['image']
                
                # Convert path to action format for the expected answer
                path = convert_path_to_actions(path)
                expected_answer = "\\boxed{Yes}" if is_gt else "\\boxed{No}"

                # Replace placeholders in the prompt
                prompt = ACTION_SAFE_PROMPT_SHORT
                # prompt = ACTION_PLAN_PROMPT.replace('<IMAGE-1>', image1_path)
                # prompt = prompt.replace('<IMAGE-2>', image2_path)
                # prompt = prompt.replace('<TEST-IMAGE>', test_image_path)
                
                # Create dataset entry
                dataset_entry = {
                    'id': maze_id,
                    'instruction': prompt.strip(),
                    'output': expected_answer,
                    'path': path,
                    'maze_size': data['size'],
                    'maze_map': data['map'],
                    'coordinates': data['coordinates'],
                    'images': {
                        'system_figure_1': image1_path,
                        'system_figure_2': image2_path,
                        'maze_image': test_image_path,
                        'vis_path_image': path_image_path
                    },
                    # 'metadata': {
                    #     'random_path': data['path_info']['random_path'],
                    #     'random_path_safe': data['path_info']['random_path_safe']
                    # }
                }
                
                dataset_entries.append(dataset_entry)
    
    # Save the dataset
    with open(output_file_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {len(dataset_entries)} dataset entries and saved to {output_file_path}")
    return dataset_entries


def generate_cot_action_plan_dataset(
        jsonl_file_path: str, output_file_path: str,
        image1_path: str = "./assets/system-figure-1.png",
        image2_path: str = "./assets/system-figure-2.png"):
    """
    Generate action plan dataset with visual Chain-of-Thought using path images.
    """
    
    dataset_entries = []
    
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            for is_gt in [True, False]:
                # Extract information from the maze data
                maze_id = data['id']+ f"_{'gt' if is_gt else 'random'}"
            
                # Extract information
                maze_id = data['id']
                if is_gt:
                    path = data['path_info']['gt_path']
                    path_image_path = data['image_paths']['path_image']
                else:
                    path = data['path_info']['random_path']
                    path_image_path = data['image_paths']['random_path_image']
                test_image_path = data['image_paths']['image']
                
                # Convert path to action format for the expected answer
                path = convert_path_to_actions(path)
                expected_answer = "\\boxed{Yes}" if is_gt else "\\boxed{No}"
                
                # Replace placeholders in the prompt
                prompt = ACTION_SAFE_PROMPT_SHORT
                cot_prompt = COT_PROMPT_TEMPLATE
                
                # Create dataset entry
                dataset_entry = {
                    'id': f"{maze_id}_cot",
                    'instruction': prompt.strip(),
                    'cot': cot_prompt.strip(),
                    'output': expected_answer,
                    'path': path,
                    'maze_size': data['size'],
                    'maze_map': data['map'],
                    'coordinates': data['coordinates'],
                    'images': {
                        'system_figure_1': image1_path,
                        'system_figure_2': image2_path,
                        'maze_image': test_image_path,
                        'vis_path_image': path_image_path
                    },
                    # 'type': 'visual_cot',
                    # 'metadata': {
                    #     'random_path': data['path_info']['random_path'],
                    #     'random_path_safe': data['path_info']['random_path_safe']
                    # }
                }
                
                dataset_entries.append(dataset_entry)
    
    # Save the CoT dataset
    with open(output_file_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {len(dataset_entries)} CoT dataset entries and saved to {output_file_path}")
    return dataset_entries


def validate_dataset(file_path: str):
    """Validate the generated dataset and return statistics."""
    entries = []
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            entries.append(entry)
    
    # Calculate statistics
    stats = {
        'total_entries': len(entries),
        'size_distribution': {},
        'average_path_length': 0,
        'sample_entry': entries[0] if entries else None
    }
    
    path_lengths = []
    for entry in entries:
        size = entry['maze_size']
        path_length = len(entry['path'])
        
        if size not in stats['size_distribution']:
            stats['size_distribution'][size] = 0
        stats['size_distribution'][size] += 1
        path_lengths.append(path_length)
    
    if path_lengths:
        stats['average_path_length'] = sum(path_lengths) / len(path_lengths)
    
    return stats


def create_sample_outputs(dataset_file: str, num_samples: int = 5):
    """Create sample outputs to demonstrate the expected format."""
    
    with open(dataset_file, 'r') as f:
        entries = [json.loads(line.strip()) for line in f]
    
    samples = entries[:num_samples]
    
    print("=" * 80)
    print("SAMPLE DATASET ENTRIES")
    print("=" * 80)
    
    for i, entry in enumerate(samples, 1):
        print(f"\n--- SAMPLE {i} ---")
        print(f"ID: {entry['id']}")
        print(f"Maze Size: {entry['maze_size']}x{entry['maze_size']}")
        print(f"Path: {entry['path']}")
        print(f"Expected Output: {entry['output']}")
        print(f"Maze Map: {entry['maze_map']}")
        print(f"Images Used:")
        for img_type, img_path in entry['images'].items():
            print(f"  - {img_type}: {img_path}")


def main():
    """Main execution function for automatic dataset generation."""
    # Configuration
    input_jsonl = "./frozen_lake_dataset_new/dataset.jsonl"
    output_dir = "./frozen_lake_dataset_new"
    
    # Output file paths
    regular_output = os.path.join(output_dir, "action_plan_dataset.jsonl")
    cot_output = os.path.join(output_dir, "action_plan_cot_dataset.jsonl")
    
    # Asset paths
    figure1_path = "./assets/system-figure-1.png"
    figure2_path = "./assets/system-figure-2.png"
    
    print("Starting automatic dataset generation...")
    
    # Generate regular action plan dataset
    print("\n1. Generating regular action plan dataset...")
    regular_entries = generate_action_plan_dataset(
        input_jsonl, regular_output, figure1_path, figure2_path
    )
    
    # Generate CoT action plan dataset
    print("\n2. Generating Chain-of-Thought action plan dataset...")
    cot_entries = generate_cot_action_plan_dataset(
        input_jsonl, cot_output, figure1_path, figure2_path
    )
    
    # Validate datasets
    print("\n3. Validating datasets...")
    regular_stats = validate_dataset(regular_output)
    cot_stats = validate_dataset(cot_output)
    
    print(f"\nRegular Dataset Stats:")
    print(f"  Total entries: {regular_stats['total_entries']}")
    print(f"  Size distribution: {regular_stats['size_distribution']}")
    print(f"  Average path length: {regular_stats['average_path_length']:.2f}")
    
    print(f"\nCoT Dataset Stats:")
    print(f"  Total entries: {cot_stats['total_entries']}")
    print(f"  Size distribution: {cot_stats['size_distribution']}")
    print(f"  Average path length: {cot_stats['average_path_length']:.2f}")
    
    # Create sample outputs
    print("\n4. Creating sample outputs...")
    create_sample_outputs(regular_output, 3)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Regular dataset: {regular_output}")
    print(f"📁 CoT dataset: {cot_output}")


# add a function to push to hf

def load_image_as_pil(image_path: str) -> PILImage.Image:
    """Load an image file as PIL Image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return PILImage.open(image_path).convert("RGB")


def prepare_hf_dataset_records(jsonl_file_path: str) -> List[Dict[str, Any]]:
    """
    Prepare records for Hugging Face dataset with proper image handling.
    
    Args:
        jsonl_file_path: Path to the JSONL file containing dataset entries
        image1_path: Path to system figure 1
        image2_path: Path to system figure 2
    
    Returns:
        List of records ready for HF dataset creation
    """
    hf_records = []
    
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            # Prepare images list in the order they appear in the instruction
            images = []
            
            # Add the maze image (appears as <TEST-IMAGE>)
            maze_image_path = entry['images']['maze_image']
            if os.path.exists(maze_image_path):
                images.append(load_image_as_pil(maze_image_path))
            
            # For CoT entries, add the solution path image
            if 'cot' in entry:
                path_image_path = entry['images']['vis_path_image']
                if os.path.exists(path_image_path):
                    images.append(load_image_as_pil(path_image_path))
            
            # Replace placeholders in instruction with numbered placeholders
            instruction = entry['instruction']
            instruction = instruction.replace('<IMAGE-1>', '')
            instruction = instruction.replace('<IMAGE-2>', '')
            instruction = instruction.replace('<TEST-IMAGE>', '<image>')
            instruction = instruction.replace('<ACTION-SEQ>', entry['path'])

            long_answer = ""
            
            # For CoT entries, handle the path image placeholder
            if 'cot' in entry:
                cot_text = entry['cot']
                cot_text = cot_text.replace('<PATH-IMAGE>', '<image_output>')
                # Combine instruction and CoT
                long_answer = f"{entry['cot'].replace('<PATH-IMAGE>', '<im_gen_start><im_gen_end>')}\n\n{entry['output']}"
            else:
                long_answer = instruction
                long_answer = entry['output']
            
            hf_record = {
                'question_id': entry['id'],
                'question': instruction,
                # 'images': images,
                'answer': entry['output'],
                'label': long_answer,  # Assuming 'output' is the expected answer
                'image': images[0] if images else None,
                'image_output': images[1] if len(images) > 1 else None,
                'path': entry['path'],
                'maze_size': entry['maze_size'],
                'maze_map': entry['maze_map']
            }
            
            hf_records.append(hf_record)
    
    return hf_records


def push_dataset_to_hf(
        records: List[Dict[str, Any]], repo_id: str,
        token: str = None, private: bool = False):
    """
    Push dataset to Hugging Face Hub.
    
    Args:
        records: List of dataset records
        repo_id: Repository ID on Hugging Face (e.g., "username/dataset-name")
        token: HF token for authentication
        private: Whether to create a private repository
    """
    # Define dataset features
    features = Features({
        'question_id': Value('string'),
        'question': Value('string'),
        # 'images': Sequence(Image()),
        'image': Image(),
        'image_output': Image(),
        'label': Value('string'),
        'answer': Value('string'),
        'path': Value('string'),
        'maze_size': Value('int32'),
        'maze_map': Sequence(Value('string'))
    })
    
    # Create dataset
    dataset = Dataset.from_list(records, features=features)
    
    # Create repository if it doesn't exist
    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
            token=token
        )
        print(f"✓ Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")
    
    # Push dataset to hub
    try:
        dataset.push_to_hub(repo_id, token=token, private=private)
        print(f"✅ Successfully pushed {len(records)} examples to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to push dataset: {e}")
        raise



HF_TOKEN = os.getenv("HF_TOKEN") or "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"

def push_both_datasets_to_hf(
        regular_dataset_path: str, cot_dataset_path: str,
        base_repo_name: str, username: str = None,
        token: str = None, private: bool = False):
    """
    Push both regular and CoT datasets to Hugging Face.
    
    Args:
        regular_dataset_path: Path to regular dataset JSONL file
        cot_dataset_path: Path to CoT dataset JSONL file
        base_repo_name: Base name for repositories (e.g., "frozen-lake-action-plan")
        username: HF username (if None, will be inferred from token)
        token: HF token for authentication
        private: Whether to create private repositories
    """
    if username is None and token is not None:
        try:
            api = HfApi(token=token)
            username = api.whoami()['name']
        except:
            raise ValueError("Could not determine username. Please provide username parameter.")

    if username is None:
        raise ValueError("Username is required when token is not provided or invalid.")

    # Prepare regular dataset
    print("📦 Preparing regular dataset...")
    regular_records = prepare_hf_dataset_records(regular_dataset_path)
    regular_repo_id = f"{username}/{base_repo_name}"

    # Prepare CoT dataset
    print("📦 Preparing CoT dataset...")
    cot_records = prepare_hf_dataset_records(cot_dataset_path)
    cot_repo_id = f"{username}/{base_repo_name}-cot"
    
    # Push regular dataset
    print(f"🚀 Pushing regular dataset to {regular_repo_id}...")
    push_dataset_to_hf(regular_records, regular_repo_id, token, private)
    
    # Push CoT dataset
    print(f"🚀 Pushing CoT dataset to {cot_repo_id}...")
    push_dataset_to_hf(cot_records, cot_repo_id, token, private)
    
    print("✅ Both datasets pushed successfully!")
    return regular_repo_id, cot_repo_id


def main_with_hf_push():
    """Enhanced main function that includes HF push functionality."""
    # Configuration
    input_jsonl = "./frozen_lake_dataset_new/dataset.jsonl"
    output_dir = "./frozen_lake_dataset_new"
    
    # Output file paths
    regular_output = os.path.join(output_dir, "action_plan_dataset.jsonl")
    cot_output = os.path.join(output_dir, "action_plan_cot_dataset.jsonl")
    
    # Asset paths
    figure1_path = "./assets/system-figure-1.png"
    figure2_path = "./assets/system-figure-2.png"
    
    print("Starting automatic dataset generation...")
    
    # Generate regular action plan dataset
    print("\n1. Generating regular action plan dataset...")
    regular_entries = generate_action_plan_dataset(
        input_jsonl, regular_output, figure1_path, figure2_path
    )
    
    # Generate CoT action plan dataset
    print("\n2. Generating Chain-of-Thought action plan dataset...")
    cot_entries = generate_cot_action_plan_dataset(
        input_jsonl, cot_output, figure1_path, figure2_path
    )
    
    # Validate datasets
    print("\n3. Validating datasets...")
    regular_stats = validate_dataset(regular_output)
    cot_stats = validate_dataset(cot_output)
    
    print(f"\nRegular Dataset Stats:")
    print(f"  Total entries: {regular_stats['total_entries']}")
    print(f"  Size distribution: {regular_stats['size_distribution']}")
    print(f"  Average path length: {regular_stats['average_path_length']:.2f}")
    
    print(f"\nCoT Dataset Stats:")
    print(f"  Total entries: {cot_stats['total_entries']}")
    print(f"  Size distribution: {cot_stats['size_distribution']}")
    print(f"  Average path length: {cot_stats['average_path_length']:.2f}")
    
    # Create sample outputs
    print("\n4. Creating sample outputs...")
    create_sample_outputs(regular_output, 3)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Regular dataset: {regular_output}")
    print(f"📁 CoT dataset: {cot_output}")
    
    # Optional: Push to Hugging Face
    push_to_hf = input("\n🤔 Would you like to push datasets to Hugging Face? (y/n): ").strip().lower()
    
    if push_to_hf in ['y', 'yes']:
        hf_token = input("🔑 Enter your Hugging Face token (or press Enter to skip): ").strip()
        if hf_token:
            username = input("👤 Enter your HF username (or press Enter to auto-detect): ").strip() or None
            base_repo_name = input("📝 Enter base repository name (default: frozen-lake-action-plan): ").strip() or "frozen-lake-action-plan"
            private = input("🔒 Make repositories private? (y/n, default: n): ").strip().lower() in ['y', 'yes']
            
            try:
                print("\n🚀 Pushing to Hugging Face...")
                regular_repo, cot_repo = push_both_datasets_to_hf(
                    regular_output, cot_output, base_repo_name, username, hf_token, private
                )
                print(f"\n🎉 Datasets available at:")
                print(f"   Regular: https://huggingface.co/datasets/{regular_repo}")
                print(f"   CoT: https://huggingface.co/datasets/{cot_repo}")
            except Exception as e:
                print(f"❌ Failed to push to Hugging Face: {e}")
        else:
            print("⏭️  Skipping Hugging Face push.")
    else:
        print("⏭️  Skipping Hugging Face push.")


# ...existing code...





if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and optionally push action plan datasets')
    parser.add_argument('--push-to-hf', action='store_true', 
                       help='Push datasets to Hugging Face after generation')
    parser.add_argument('--hf-token', type=str, default=HF_TOKEN,
                       help='Hugging Face token for authentication')
    parser.add_argument('--hf-username', type=str, default='linjieli222',
                       help='Hugging Face username')
    parser.add_argument('--repo-name', type=str, default='frozen-lake-action-safe-single-image-tag-20k',
                       help='Base repository name (default: frozen-lake-action-safe-single-image-tag-20k)')
    parser.add_argument('--private', action='store_true',
                       help='Create private repositories')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive mode for HF push')
    
    args = parser.parse_args()
    
    if args.interactive:
        main_with_hf_push()
    else:
        main()
        
        if args.push_to_hf:
            if not args.hf_token:
                print("❌ HF token required for pushing. Use --hf-token or --interactive")
                exit(1)
            
            # Paths
            output_dir = "./frozen_lake_dataset_new"
            regular_output = os.path.join(output_dir, "action_plan_dataset.jsonl")
            cot_output = os.path.join(output_dir, "action_plan_cot_dataset.jsonl")
            
            try:
                print(f"\n🚀 Pushing datasets to Hugging Face...")
                regular_repo, cot_repo = push_both_datasets_to_hf(
                    regular_output, cot_output, 
                    args.repo_name, args.hf_username, 
                    args.hf_token, args.private
                )
                print(f"\n🎉 Datasets available at:")
                print(f"   Regular: https://huggingface.co/datasets/{regular_repo}")
                print(f"   CoT: https://huggingface.co/datasets/{cot_repo}")
            except Exception as e:
                print(f"❌ Failed to push to Hugging Face: {e}")
                exit(1)
