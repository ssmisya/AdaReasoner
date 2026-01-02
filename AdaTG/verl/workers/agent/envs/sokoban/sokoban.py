# import numpy as np
# import copy
# from typing import Optional, List

# import gymnasium as gym
# from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
# from gymnasium.utils import seeding
# from tool_envs import ToolBase, extract_tool_call_contents

# class SokobanEnv(ToolBase):
#     name = "sokoban"

#     GRID_LOOKUP = {
#         0: " # \t",  # wall
#         1: " _ \t",  # floor
#         2: " O \t",  # target
#         3: " √ \t",  # box on target
#         4: " X \t",  # box
#         5: " P \t",  # player
#         6: " S \t",  # player on target
#         # Use tab separator to separate columns and \n\n to separate rows.
#     }

#     ACTION_LOOKUP = {
#         0: "None",
#         1: "Up",
#         2: "Down",
#         3: "Left",
#         4: "Right",
#     }

#     PENALTY_FOR_INVALID = -0.5
#     ACTION_START = '<answer>'
#     ACTION_END = '</answer>'

#     def __init__(self, **kwargs):
#         super().__init__(
#             name=self.name,
#             description="A tool that simulates the Sokoban environment.",
#             parameters={
#                 "type": "object",
#                 "properties": {
#                     "action": {"type": "integer", "description": "The action to take (0: none, 1: left, 2: down, 3: right, 4: up)"}
#                 },
#                 "required": ["action"]
#             }
#         )
#         self.env = None
#         self.GRID_LOOKUP_INV = {v : k for k, v in self.GRID_LOOKUP.items()}
#         self.ACTION_LOOKUP_INV = {v : k for k, v in self.ACTION_LOOKUP.items()}

#     def execute(self, action_string) -> str:
#         """
#         Execute the tool functionality by taking an action in the Sokoban environment.
        
#         Args:
#             action: The action to take (0: none, 1: left, 2: down, 3: right, 4: up)
            
#         Returns:
#             observation: The current state of the environment
#             reward: The reward received after taking the action
#         """
#         action = kwargs.get("action", 0)
        
#         player_pos, reward, done, _, _ = self.env.step(action)
#         self.reward = reward
#         print(player_pos, reward, done)
#         if done:
#             self.reset()
        
#         return self.env.render(), self.reward

#     def reset(self, raw_prompt):
#         """
#         Reset the environment to its initial state.
#         """
#         if self.env is not None:
#             self.env.close()
#             self.env = None

#         user_prompt = raw_prompt[0]['content']
#         init_map = extract_sokoban_map_from_input_prompt(user_prompt)
#         self.env = 
#         self.reward = 0


# # Helper function to generate a random map for the Frozen Lake environment
# def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
#     """Generates a random valid map (one that has a path from start to goal)

#     Args:
#         size: size of each side of the grid
#         p: probability that a tile is frozen
#         seed: optional seed to ensure the generation of reproducible maps

#     Returns:
#         A random valid map
#     """
#     valid = False
#     board = []  # initialize to make pyright happy

#     np_random, _ = seeding.np_random(seed)

#     # generate random start and end points

#     while not valid:
#         p = min(1, p)
#         board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])

#         while True:
#             start_r = np_random.integers(0, size)
#             start_c = np_random.integers(0, size)
#             goal_r = np_random.integers(0, size)
#             goal_c = np_random.integers(0, size)
            
#             # Ensure start and goal are different positions
#             if (start_r, start_c) != (goal_r, goal_c):
#                 break
            
#         board[start_r][start_c] = "S"
#         board[goal_r][goal_c] = "G"
        
#         valid = is_valid(board, size)
#     return ["".join(x) for x in board]


# # Example usage
# # action:[int]
# # LEFT = 0
# # DOWN = 1
# # RIGHT = 2
# # UP = 3
# if __name__ == "__main__":
#     tool = FrozenLakeTool()
#     observation, reward = tool.execute(action=1)
#     print(f"Observation:\n{observation}")
#     print(f"Reward: {reward}")
#     observation, reward = tool.execute(action=2)
#     print(f"Observation:\n{observation}")
#     print(f"Reward: {reward}")