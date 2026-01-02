ACTION_PLAN_PROMPT = """"
As a professional maze solver, your task is to analyze a grid-based map and devise an action plan that enables a player to reach the goal from the starting point without falling into any holes, using the fewest possible moves. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

## Game Setup
- The game presents a fully observable grid-based map.
- The player starts at a specified grid square, with the goal located elsewhere on the map.
- Each grid square is either safe or contains a hole.
- Your goal is to guide the player to the goal while avoiding holes.
The following figure shows how the player, the holes (non-safe grid), the lands (safe grids), and the goals look like.

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
Now you will solve the given maze. To solve it, please generate text EXACTLY FOLLOW THE FOLLOWING STEPS:
1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.
2. Then, generate an action plan to navigate to the goal step by step. At each step, you should check:
    (a) Where the current move leads the player to (the row and column);
    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?
    (c) Determine if that is a safe action. If not, correct it and re-generate the action plan.
3. Next, verify if the steps successfully navigate the player to the goal without falling into the hole. If not, restart from step 2 and re-generate this step.
4. If succeed, output an aggregated plan using "Action plan: <PLAN>", where <PLAN> is a string concatenated action in each step. For example, "Action plan: L,L,R,U,D" meaning an action plan of left, left, right, up, and down. Double check the final action plan is consistent with the previous analysis.
Do not output any extra content after the above aggregated output.

Please generate action plan for the following maze:

<TEST-IMAGE>
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
Here is a potential solution to the maze above, the solution path is shown in the image below with arrows indicating the direction of movement:

<PATH-IMAGE>
"""
    
def convert_path_to_actions(path_string: str) -> str:
    """Convert path string (e.g., 'ull') to action format (e.g., 'U,L,L')"""
    action_map = {'u': 'U', 'd': 'D', 'l': 'L', 'r': 'R'}
    actions = [action_map[char.lower()] for char in path_string if char.lower() in action_map]
    return ','.join(actions)



def generate_action_plan_dataset(jsonl_file_path: str, output_file_path: str,
                                image1_path: str = "./assets/system-figure-1.png",
                                image2_path: str = "./assets/system-figure-2.png"):
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
            
            # Extract information from the maze data
            maze_id = data['id']
            gt_path = data['path_info']['gt_path']
            test_image_path = data['image_paths']['image']
            path_image_path = data['image_paths']['path_image']
            
            # Convert path to action format for the expected answer
            expected_actions = convert_path_to_actions(gt_path)
            expected_answer = f"Action plan: {expected_actions}"
            
            # Replace placeholders in the prompt
            prompt = ACTION_PLAN_PROMPT
            # prompt = ACTION_PLAN_PROMPT.replace('<IMAGE-1>', image1_path)
            # prompt = prompt.replace('<IMAGE-2>', image2_path)
            # prompt = prompt.replace('<TEST-IMAGE>', test_image_path)
            
            # Create dataset entry
            dataset_entry = {
                'id': maze_id,
                'instruction': prompt.strip(),
                'output': expected_answer,
                'gt_path': gt_path,
                'maze_size': data['size'],
                'maze_map': data['map'],
                'coordinates': data['coordinates'],
                'images': {
                    'system_figure_1': image1_path,
                    'system_figure_2': image2_path,
                    'maze_image': test_image_path,
                    'solution_path': path_image_path
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


def generate_cot_action_plan_dataset(jsonl_file_path: str, output_file_path: str,
                                    image1_path: str = "./assets/system-figure-1.png",
                                    image2_path: str = "./assets/system-figure-2.png"):
    """
    Generate action plan dataset with visual Chain-of-Thought using path images.
    """
    
    dataset_entries = []
    
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract information
            maze_id = data['id']
            gt_path = data['path_info']['gt_path']
            test_image_path = data['image_paths']['image']
            path_image_path = data['image_paths']['path_image']
            
            # Convert path to action format
            expected_actions = convert_path_to_actions(gt_path)
            expected_answer = f"Action plan: {expected_actions}"
            
            # Replace placeholders in the CoT prompt
            prompt = ACTION_PLAN_PROMPT
            cot_prompt = COT_PROMPT_TEMPLATE
            
            # Create dataset entry
            dataset_entry = {
                'id': f"{maze_id}_cot",
                'instruction': prompt.strip(),
                'cot': cot_prompt.strip(),
                'output': expected_answer,
                'gt_path': gt_path,
                'maze_size': data['size'],
                'maze_map': data['map'],
                'coordinates': data['coordinates'],
                'images': {
                    'system_figure_1': image1_path,
                    'system_figure_2': image2_path,
                    'maze_image': test_image_path,
                    'solution_path': path_image_path
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
        path_length = len(entry['gt_path'])
        
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
        print(f"GT Path: {entry['gt_path']}")
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


def prepare_hf_dataset_records(jsonl_file_path: str, 
                              image1_path: str = "./assets/system-figure-1.png",
                              image2_path: str = "./assets/system-figure-2.png") -> List[Dict[str, Any]]:
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
            
            # Add system figures first (they appear in the instruction as <IMAGE-1> and <IMAGE-2>)
            if os.path.exists(image1_path):
                images.append(load_image_as_pil(image1_path))
            if os.path.exists(image2_path):
                images.append(load_image_as_pil(image2_path))
            
            # Add the maze image (appears as <TEST-IMAGE>)
            maze_image_path = entry['images']['maze_image']
            if os.path.exists(maze_image_path):
                images.append(load_image_as_pil(maze_image_path))
            
            # For CoT entries, add the solution path image
            if 'cot' in entry:
                solution_path_image = entry['images']['solution_path']
                if os.path.exists(solution_path_image):
                    images.append(load_image_as_pil(solution_path_image))
            
            # Replace placeholders in instruction with numbered placeholders
            instruction = entry['instruction']
            instruction = instruction.replace('<IMAGE-1>', '<image_1>')
            instruction = instruction.replace('<IMAGE-2>', '<image_2>')
            instruction = instruction.replace('<TEST-IMAGE>', '<image_3>')
            
            # For CoT entries, handle the path image placeholder
            if 'cot' in entry:
                cot_text = entry['cot']
                cot_text = cot_text.replace('<PATH-IMAGE>', '<image_4>')
                # Combine instruction and CoT
                full_instruction = instruction + "\n\n" + cot_text
            else:
                full_instruction = instruction
            
            hf_record = {
                'question_id': entry['id'],
                'question': full_instruction,
                'images': images,
                'answer': entry['output'],
                'gt_path': entry['gt_path'],
                'maze_size': entry['maze_size'],
                'maze_map': entry['maze_map']
            }
            
            hf_records.append(hf_record)
    
    return hf_records


def push_dataset_to_hf(records: List[Dict[str, Any]], repo_id: str, 
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
        'images': Sequence(Image()),
        'answer': Value('string'),
        'gt_path': Value('string'),
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

def push_both_datasets_to_hf(regular_dataset_path: str, cot_dataset_path: str,
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
    parser.add_argument('--repo-name', type=str, default='frozen-lake-action-plan',
                       help='Base repository name (default: frozen-lake-action-plan)')
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
