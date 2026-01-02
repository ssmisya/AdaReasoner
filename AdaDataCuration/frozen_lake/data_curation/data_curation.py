import os,sys,json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a_star_calc import A_StarCalculatir
from PIL import Image
import gymnasium as gym
import re
from gymnasium.envs.toy_text.frozen_lake import is_valid
from gymnasium.envs.toy_text.frozen_lake import generate_random_map as default_generate_random_map
from PIL import Image, ImageDraw
import heapq
import os
import numpy as np
import random
from tqdm import tqdm
from typing import Optional, List
from gymnasium.utils import seeding


def check_valid_map(text_map):
    frozen_lake_default_valid = is_valid(text_map, len(text_map))
    # add check for start is not a hole and goal is not a hole and goal is not a start
    startpos = None
    goalpos = None
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            if cell == 'S':
                startpos = (i, j)
            elif cell == 'G':
                goalpos = (i, j)
    if startpos is None or goalpos is None:
        raise ValueError("Map must contain a start point 'S' and a goal point 'G'")
    if startpos == goalpos:
        return False
    if text_map[startpos[0]][startpos[1]] == 'H':
        return False
    if text_map[goalpos[0]][goalpos[1]] == 'H':
        return False
    return frozen_lake_default_valid


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None,
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        
        # Randomly place start point
        start_row = np_random.integers(0, size)
        start_col = np_random.integers(0, size)
        board[start_row][start_col] = "S"

        goal_row = np_random.integers(0, size)
        goal_col = np_random.integers(0, size)
        board[goal_row][goal_col] = "G"

        valid = check_valid_map(board)
    return ["".join(x) for x in board]


def draw_direction_sequence(image, start, directions, step=64, linewidth=3):
    """
    在图片上从起点沿方向序列画线段。
    
    Args:
        image_path (str): 图片文件路径
        start (tuple): 起点坐标 (x, y)
        directions (str): 方向序列, 由 'u','d','l','r' 组成
        step (int): 每个方向移动的像素数
        save_path (str): 保存输出文件名
    """
    # 打开图片
    if isinstance(image,str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError("image must be a file path or a PIL Image object")
    draw = ImageDraw.Draw(img)

    # 当前坐标
    x, y = start

    # 遍历方向
    for dir in directions:
        old_x, old_y = x, y
        if dir == 'u':
            y -= step
        elif dir == 'd':
            y += step
        elif dir == 'l':
            x -= step
        elif dir == 'r':
            x += step
        else:
            raise ValueError(f"未知方向: {dir}")
        
        # 画从(old_x, old_y)到(x, y)的线
        draw.line([(old_x, old_y), (x, y)], fill="red", width=linewidth)
        
        # Add arrow head for each line segment
        arrow_size = linewidth * 2
        # Calculate arrow head points
        if dir == 'u':
            # Arrow pointing up
            arrow_points = [
                (x, y),
                (x - arrow_size, y + arrow_size * 2),
                (x + arrow_size, y + arrow_size * 2)
            ]
        elif dir == 'd':
            # Arrow pointing down
            arrow_points = [
                (x, y),
                (x - arrow_size, y - arrow_size * 2),
                (x + arrow_size, y - arrow_size * 2)
            ]
        elif dir == 'l':
            # Arrow pointing left
            arrow_points = [
                (x, y),
                (x + arrow_size * 2, y - arrow_size),
                (x + arrow_size * 2, y + arrow_size)
            ]
        elif dir == 'r':
            # Arrow pointing right
            arrow_points = [
                (x, y),
                (x - arrow_size * 2, y - arrow_size),
                (x - arrow_size * 2, y + arrow_size)
            ]
        # Draw the arrow head
        draw.polygon(arrow_points, fill="red")

    return img

def draw_route(image, start_point, directions, pixel_coordinate=False, step = 64):
    half_length = step/2
    match = re.match(r'\[\s*([\d\.,\s]+)\s*\]', start_point)
    if match:
        try:
            # Extract coordinates
            coords_str = match.group(1).split(',')
            # Try to convert to float
            crop_coords = [float(c) for c in coords_str]
            for coord in crop_coords:
                if not isinstance(coord, (int, float)):
                    raise ValueError("Coordinates must be numeric.")
                elif coord < 1:
                    raise ValueError("Coordinates must be greater than or equal to 1.")
                
            assert len(crop_coords) == 2, "Start point must have exactly two coordinates."
        except ValueError:
            return -1
        
    if not pixel_coordinate:
        crop_coords = [(item-1) * step + half_length for item in crop_coords]
        
    edited_image = draw_direction_sequence(image, tuple(crop_coords), directions, step=step)
    return edited_image

def extract_coordinates(text_map, cell_size=64):
    """
    Extract pixel coordinates of start point (S), holes (H), and goal (G) from a text map.
    
    Args:
        text_map (list of lists): 2D list representing the text map
        cell_size (int): Size of each cell in pixels
    
    Returns:
        dict: Dictionary containing pixel coordinates for 'start', 'holes', and 'goal'
    """
    half_cell = cell_size / 2
    start_point = None
    holes = []
    goal_point = None
    
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            # Calculate pixel coordinates (center of the cell)
            pixel_x = j * cell_size + half_cell
            pixel_y = i * cell_size + half_cell
            
            if cell == 'S':
                start_point = (pixel_x, pixel_y)
            elif cell == 'H':
                holes.append((pixel_x, pixel_y))
            elif cell == 'G':
                goal_point = (pixel_x, pixel_y)
    
    return {
        'start': start_point,
        'holes': holes,
        'goal': goal_point
    }


def will_fall_into_hole(text_map, path_string, start_point=None):
    """
    Check if following a given path will cause the agent to fall into a hole.
    
    Args:
        text_map (list): 2D list representing the FrozenLake map
        path_string (str): String of actions (r, l, u, d)
        start_point (tuple): Starting position as (row, col) coordinates.
                           If None, finds 'S' in map.
    
    Returns:
        bool: True if the path leads to a hole, False otherwise
    """
    # Find start point if not provided
    if start_point is None:
        start_point = None
        for i, row in enumerate(text_map):
            for j, cell in enumerate(row):
                if cell == 'S':
                    start_point = (i, j)
                    break
            if start_point:
                break
        
        if start_point is None:
            raise ValueError("No start point 'S' found in the map")
    
    map_height = len(text_map)
    map_width = len(text_map[0]) if text_map else 0
    
    # Current position
    row, col = start_point
    
    # Check if starting position is valid
    if not (0 <= row < map_height and 0 <= col < map_width):
        return True  # Invalid start position is considered falling into hole
    
    # Simulate the path
    for direction in path_string:
        if direction == 'u':
            row -= 1
        elif direction == 'd':
            row += 1
        elif direction == 'l':
            col -= 1
        elif direction == 'r':
            col += 1
        else:
            # Invalid direction
            continue
        
        # Check if current position is within bounds
        if not (0 <= row < map_height and 0 <= col < map_width):
            return True  # Out of bounds is considered falling into hole
        
        # Check if current position is a hole
        if text_map[row][col] == 'H':
            return True
    
    return False


def astar_search(start, goal, holes, cell_size=64):
    """
    A* search algorithm to find the shortest path from start to goal while avoiding holes.
    
    Args:
        start (tuple): Start position (pixel coordinates)
        goal (tuple): Goal position (pixel coordinates)
        holes (list): List of hole positions (pixel coordinates)
        cell_size (int): Size of each cell in pixels
    
    Returns:
        str: String representing the path as directions ('l', 'r', 'u', 'd')
    """
    # Convert pixel coordinates to grid coordinates (0-based)
    def pixel_to_grid(pixel_coord):
        x, y = pixel_coord
        grid_x = int(x // cell_size)
        grid_y = int(y // cell_size)
        return (grid_x, grid_y)
    
    # Calculate grid size from coordinates
    all_points = [start, goal] + holes
    max_x = max(point[0] for point in all_points) / cell_size
    max_y = max(point[1] for point in all_points) / cell_size
    grid_size = max(int(max_x), int(max_y)) + 1
    
    start_grid = pixel_to_grid(start)
    goal_grid = pixel_to_grid(goal)
    holes_grid = [pixel_to_grid(hole) for hole in holes]
    
    # Define possible moves: left, right, up, down
    # Changed to match coordinate system where left is -x and up is -y
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction_chars = ['l', 'r', 'u', 'd']
    
    # Initialize data structures for A*
    open_set = []
    closed_set = set()
    g_score = {start_grid: 0}
    f_score = {start_grid: manhattan_distance(start_grid, goal_grid)}
    came_from = {}
    
    # Push start node to priority queue
    heapq.heappush(open_set, (f_score[start_grid], start_grid))
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_grid:
            return reconstruct_path(came_from, current, direction_chars)
            
        closed_set.add(current)
        
        for i, (dx, dy) in enumerate(directions):
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if the neighbor is valid
            if (not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size) or
                neighbor in holes_grid or neighbor in closed_set):
                continue
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = (current, i)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal_grid)
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current, direction_chars):
    """Reconstruct the path from start to goal using the came_from dictionary"""
    path = []
    while current in came_from:
        current, direction_idx = came_from[current]
        path.append(direction_chars[direction_idx])
    
    # Reverse the path and convert to string
    return ''.join(path[::-1])

def find_shortest_safe_path(locs):
    """Find the shortest safe path from start to goal avoiding holes"""
    start = locs['start']
    goal = locs['goal']
    holes = locs['holes']
    
    path = astar_search(start, goal, holes)
    return path


def verify_path(text_map, path_string, verbose=True):
    """
    Verify if a given sequence of actions leads to safely reaching the goal in the FrozenLake environment.
    
    Args:
        text_map (list): 2D list representing the FrozenLake map
        path_string (str): String of actions (r, l, u, d)
        verbose (bool): Whether to print detailed information during verification
    
    Returns:
        bool: True if the path leads to the goal, False otherwise
    """
    # Create environment
    env = gym.make('FrozenLake-v1', desc=text_map, render_mode="rgb_array" if verbose else None, is_slippery=False)
    obs, _ = env.reset()
    
    # Dictionary to map directions to actions expected by FrozenLake
    direction_to_action = {
        'l': 0,  # LEFT
        'd': 1,  # DOWN
        'r': 2,  # RIGHT
        'u': 3,  # UP
    }
    
    # Follow the path
    success = True
    step_count = 0
    
    for direction in path_string:
        action = direction_to_action.get(direction.lower())
        if action is None:
            if verbose:
                print(f"Invalid direction: {direction}")
            success = False
            break
            
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if verbose:
            print(f"Step {step_count}: Action {direction} -> Position {obs}, Reward {reward}")
            # Render the current state
            rgb_matrix = env.render()
            image = Image.fromarray(rgb_matrix)
            # display(image)
        
        if terminated:
            # Check if we reached the goal or fell into a hole
            if reward == 1:
                if verbose:
                    print("Success! Reached the goal.")
            else:
                if verbose:
                    print("Failed! Fell into a hole.")
                success = False
            break
    
    # Check if we reached the goal
    if not terminated and verbose:
        print("Path completed but did not reach the goal.")
        success = False
    
    env.close()
    return success and reward == 1

def generate_random_path(text_map, min_length=5, max_length=20, within_map=True, start_point=None):
    """
    Generate a random path for a FrozenLake environment.
    The path may not be valid and is generated randomly.
    
    Args:
        text_map (list): 2D list representing the FrozenLake map
        min_length (int): Minimum length of the path
        max_length (int): Maximum length of the path
    
    Returns:
        str: String representing the random path as directions ('l', 'r', 'u', 'd')
    """
    # Define possible moves
    directions = ['l', 'r', 'u', 'd']
    
    # Determine path length
    map_size = len(text_map)
    if max_length < min_length:
        max_length = min_length
    
    # Adjust max_length based on map size if needed
    suggested_max = map_size * 4  # Reasonable upper limit based on map size
    if max_length > suggested_max:
        max_length = suggested_max
    
    # Generate a random path length
    path_length = random.randint(min_length, max_length)
    
    # Generate random directions
    path = ''.join(random.choices(directions, k=path_length))

    # add a check to ensure the path is within the map boundaries
    # if not regenerate
    if within_map:
        # check start point, if None, raise ValueError
        if start_point is None:
            # find the start point in the text_map
            start_point = next((i, j) for i, row in enumerate(text_map) for j, cell in enumerate(row) if cell == 'S')
        while not is_path_within_map(path, text_map, start_point):
            path = ''.join(random.choices(directions, k=path_length))
    
    # # also truncate the path as soon as it reaches the goal or falls into a hole
    # if start_point is None:
    #     start_point = next((i, j) for i, row in enumerate(text_map) for j, cell in enumerate(row) if cell == 'S')
    # row, col = start_point
    # for direction in path:
    #     if direction == 'u':
    #         row -= 1
    #     elif direction == 'd':
    #         row += 1
    #     elif direction == 'l':
    #         col -= 1
    #     elif direction == 'r':
    #         col += 1
        
    #     # Check if current position is within bounds
    #     if not (0 <= row < len(text_map) and 0 <= col < len(text_map[0])):
    #         break
        
    #     # Check if current position is a hole
    #     if text_map[row][col] == 'H' or text_map[row][col] == 'G':
    #         path = path[:path.index(direction)+1]
    #         break

    return path


def is_path_within_map(path, text_map, start_point):
    """
    Check if a given path stays within the map boundaries.
    
    Args:
        path (str): String of directions ('l', 'r', 'u', 'd')
        text_map (list): 2D list representing the FrozenLake map
        start_point (tuple): Starting position as (row, col) coordinates
    
    Returns:
        bool: True if the path stays within map boundaries, False otherwise
    """
    map_height = len(text_map)
    map_width = len(text_map[0]) if text_map else 0
    
    # Current position
    row, col = start_point
    
    # Check if starting position is valid
    if not (0 <= row < map_height and 0 <= col < map_width):
        return False
    
    # Simulate the path
    for direction in path:
        if direction == 'u':
            row -= 1
        elif direction == 'd':
            row += 1
        elif direction == 'l':
            col -= 1
        elif direction == 'r':
            col += 1
        else:
            # Invalid direction
            return False
        
        # Check if current position is within bounds
        if not (0 <= row < map_height and 0 <= col < map_width):
            return False
    
    return True

# instead of generating a random path, we will randomly perturb the correct path
# and then check if it is still valid or not

def perturb_path(path, text_map, perturbation_rate=0.2):
    """
    Perturb a given path by randomly changing some of its directions.
    
    Args:
        path (str): Original path as a string of directions ('l', 'r', 'u', 'd')
        perturbation_rate (float): Probability of changing each direction in the path
    
    Returns:
        str: Perturbed path
    """
    perturbed_path = []
    for direction in path:
        if random.random() < perturbation_rate:
            # Randomly change the direction
            perturbed_direction = random.choice(['l', 'r', 'u', 'd'])
            perturbed_path.append(perturbed_direction)
        else:
            perturbed_path.append(direction)

    # make sure the perturbed path falls within the map boundaries
    # if not, regenerate the path
    startpoint = next((i, j) for i, row in enumerate(text_map) for j, cell in enumerate(row) if cell == 'S')
    while not is_path_within_map(''.join(perturbed_path), text_map, startpoint):
        return perturb_path(path, text_map, perturbation_rate)
    
    return ''.join(perturbed_path)


def generate_frozen_lake_dataset(save_dir, size_range=(5, 8), samples_per_size=200):
    """
    Generate and save FrozenLake environments with varying sizes.
    
    Args:
        save_dir (str): Directory to save the generated images
        size_range (tuple): Range of sizes (min, max) inclusive
        samples_per_size (int): Number of samples to generate for each size
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a subdirectory for images
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create a subdirectory for metadata
    meta_dir = os.path.join(save_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    
    min_size, max_size = size_range
    
    # Create a JSONL file for the dataset
    jsonl_path = os.path.join(save_dir, "dataset.jsonl")
    
    # Generate environments for each size
    with open(jsonl_path, 'w') as jsonl_file:
        for size in range(min_size, max_size + 1):
            print(f"Generating {samples_per_size} FrozenLake environments of size {size}x{size}")
            current_image_dir = os.path.join(img_dir, f"size_{size}")
            os.makedirs(current_image_dir, exist_ok=True)

            # add a set() for checking whether the map is already generated
            generated_maps = set()

            # first check the maximum number of possible unique maps and adjust samples_per_size accordingly
            # More accurate calculation: choose 2 positions for S and G, rest can be F or H
            total_cells = size * size
            # Choose 2 positions for S and G: C(total_cells, 2)
            sg_combinations = total_cells * (total_cells - 1)  # ordered pairs for S and G positions
            # Remaining cells can be F or H: 2^(total_cells - 2)
            remaining_combinations = 2 ** (total_cells - 2)
            max_unique_maps = sg_combinations * remaining_combinations

            # For very large sizes, this becomes astronomically large, so cap it at a reasonable number
            reasonable_max = min(max_unique_maps, 1000000)  # Cap at 1 million unique maps

            num_continued_skips = 0
            
            if samples_per_size > reasonable_max:
                print(f"Warning: samples_per_size ({samples_per_size}) exceeds maximum unique maps ({reasonable_max}) for size {size}. Adjusting samples_per_size to {reasonable_max}.")
                samples_per_size = reasonable_max
            sample_idx = 1
            pbar = tqdm(total=samples_per_size, desc=f"Generating size {size}x{size} environments", unit="sample")
            while sample_idx < samples_per_size + 1:
                # Generate a unique ID for this environment
                env_id = f"frozenlake_s{size}_{sample_idx}"

                random_startpo = random.choice([True, False])  # Randomly decide whether to generate a random start point
                
                # Generate a random map
                try:
                    if not random_startpo:
                        # Generate a valid map with a fixed start and goal
                        text_map = default_generate_random_map(size=size, p=0.8)
                    else:
                        text_map = generate_random_map(size=size, p=0.8)  # p=0.8 means 20% chance of holes

                    # serialize the map to a string to check for uniqueness
                    text_map_str = '\n'.join(text_map)
                    if text_map_str in generated_maps:
                        print(f"Map already generated for {env_id}, skipping...")
                        num_continued_skips += 1
                        if num_continued_skips > 100:
                            print(f"Too many continued skips ({num_continued_skips}), breaking out of the loop.")
                            break
                        continue
                    else:
                        num_continued_skips = 0

                    startpos = next((i, j) for i, row in enumerate(text_map) for j, cell in enumerate(row) if cell == 'S')
                    
                    # Create environment
                    env = gym.make('FrozenLake-v1', desc=text_map, render_mode="rgb_array")
                    env.reset()
                    rgb_matrix = env.render()
                    image = Image.fromarray(rgb_matrix)
                    
                    # Extract coordinates
                    locs = extract_coordinates(text_map)
                    print(f"Extracted coordinates: {locs}")
                    
                    # Find the shortest path
                    path = find_shortest_safe_path(locs)
                    assert verify_path(text_map, path, verbose=False), "Path verification failed"
                    # Only save if a valid path exists
                    if path:
                        # Draw the path on the image
                        image_with_path = draw_route(image, f"[{locs['start'][0]}, {locs['start'][1]}]", path, pixel_coordinate=True)
                        
                        # Save the image
                        img_path = os.path.join(current_image_dir, f"{env_id}.png")
                        image.save(img_path)
                        
                        # Save image with path for visualization
                        path_img_path = os.path.join(current_image_dir, f"{env_id}_path.png")
                        image_with_path.save(path_img_path)
                        
                        random_path_safe = True
                        num_tries = 0

                        while random_path_safe and num_tries < 10:
                            # random_path = generate_random_path(text_map)
                            random_path = perturb_path(path, text_map, perturbation_rate=0.4)


                            random_path_safe = not will_fall_into_hole(text_map, random_path)
                            num_tries += 1
                        if not random_path_safe:
                            print(f"Generated random path for {env_id} that falls into a hole: {random_path}")
                            image_with_random_path = draw_route(image, f"[{locs['start'][0]}, {locs['start'][1]}]", random_path, pixel_coordinate=True)
                            random_path_img_path = os.path.join(current_image_dir, f"{env_id}_random_path.png")
                            image_with_random_path.save(random_path_img_path)
                            
                            image_paths = {
                                "image": img_path,
                                "path_image": path_img_path,
                                "random_path_image": random_path_img_path
                            }
                            path_info = {
                                "gt_path": path,
                                "random_path": random_path,
                                "random_path_safe": random_path_safe,
                            }
                            # Create metadata
                            metadata = {
                                "id": env_id,
                                "size": size,
                                "map": text_map,
                                "path_info": path_info,
                                "coordinates": locs,
                                "image_paths": image_paths,
                            }

                            # Add to the JSONL file (one JSON object per line)
                            jsonl_file.write(json.dumps(metadata) + '\n')
                            
                            print(f"Generated {env_id}")

                except Exception as e:
                    print(f"Error generating {env_id}: {e}")
                    # Try again for this sample
                
                # Close the environment
                env.close()
                    
                pbar.update(1)
                sample_idx += 1
                generated_maps.add(text_map_str)
    
    print(f"Dataset generation complete. Data stored in {jsonl_path}")
    

# Example usage
if __name__ == "__main__":
    generate_frozen_lake_dataset(size_range=(3, 8), save_dir="./frozen_lake_dataset_new", samples_per_size=2000)

