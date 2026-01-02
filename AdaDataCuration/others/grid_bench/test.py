from __future__ import annotations
import random
import matplotlib.pyplot as plt
from PIL import Image

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class CustomEmptyEnv(MiniGridEnv):
    """自定义空环境，可指定起点和终点位置"""
    
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=(8, 8),
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # 创建空网格
        self.grid = Grid(width, height)

        # 生成外围墙壁
        self.grid.wall_rect(0, 0, width, height)

        # 放置目标
        if self.goal_pos is not None:
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # 设置agent位置和方向
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "reach the goal"


def create_env_from_positions(size, agent_pos, agent_dir, goal_pos, render_mode='rgb_array'):
    """
    根据指定的起点和终点创建环境
    
    Args:
        size: 网格大小
        agent_pos: agent起始位置 (x, y)
        agent_dir: agent起始方向 (0=右, 1=下, 2=左, 3=上)
        goal_pos: 目标位置 (x, y)
        render_mode: 渲染模式
        
    Returns:
        env: 创建的环境
    """
    env = CustomEmptyEnv(
        size=size,
        agent_start_pos=agent_pos,
        agent_start_dir=agent_dir,
        goal_pos=goal_pos,
        render_mode=render_mode
    )
    env.reset()
    return env


def direction_to_action(env, direction_char):
    """将方向字符(L/R/U/D)转换为MiniGrid的动作序列"""
    current_direction = env.unwrapped.agent_dir
    
    direction_map = {'R': 0, 'D': 1, 'L': 2, 'U': 3}
    
    target_direction = direction_map.get(direction_char.upper())
    if target_direction is None:
        raise ValueError(f"Invalid direction: {direction_char}")
    
    turn_diff = (target_direction - current_direction) % 4
    
    actions = []
    
    if turn_diff == 1:
        actions.append(1)  # turn right
    elif turn_diff == 2:
        actions.append(1)
        actions.append(1)
    elif turn_diff == 3:
        actions.append(0)  # turn left
    
    actions.append(2)  # forward
    
    return actions


def test_path_with_reward(env, path_string, verbose=True, render_steps=False):
    """
    测试给定路径并返回详细信息
    
    Args:
        env: MiniGrid环境
        path_string: 逗号分隔的方向字符串 (L,R,U,D)
        verbose: 是否打印详细信息
        render_steps: 是否保存每一步的图像
        
    Returns:
        dict: 包含是否成功、总奖励、步数等信息
    """
    env.reset()
    
    directions = [d.strip() for d in path_string.split(',')]
    
    total_reward = 0
    step_count = 0
    images = []
    
    if verbose:
        print(f"Starting position: {env.agent_pos}, direction: {env.agent_dir}")
        print(f"Goal position: {env.unwrapped.goal_pos}")
        print(f"Path to test: {path_string}\n")
    
    for i, direction in enumerate(directions):
        if verbose:
            print(f"Step {i+1}: Moving {direction}")
        
        try:
            actions = direction_to_action(env, direction)
        except ValueError as e:
            if verbose:
                print(f"❌ Invalid direction: {e}")
            return {
                "success": False,
                "total_reward": total_reward,
                "steps": step_count,
                "reason": "Invalid direction",
                "images": images
            }
        
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if render_steps:
                images.append(Image.fromarray(env.render()))
            
            if verbose and reward != 0:
                print(f"  ⭐ Reward received: {reward}")
            
            if terminated:
                if verbose:
                    print(f"\n✅ SUCCESS! Reached goal!")
                    print(f"Total reward: {total_reward}")
                    print(f"Total steps: {step_count}")
                
                return {
                    "success": True,
                    "total_reward": total_reward,
                    "steps": step_count,
                    "reached_goal": True,
                    "images": images
                }
            
            if truncated:
                if verbose:
                    print(f"\n⏱️ Episode truncated (timeout)")
                    print(f"Total reward: {total_reward}")
                    print(f"Total steps: {step_count}")
                
                return {
                    "success": False,
                    "total_reward": total_reward,
                    "steps": step_count,
                    "reason": "Truncated",
                    "images": images
                }
        
        if verbose:
            print(f"  Current position: {env.agent_pos}, direction: {env.agent_dir}")
    
    if verbose:
        print(f"\n❌ Path completed but goal not reached")
        print(f"Total reward: {total_reward}")
        print(f"Total steps: {step_count}")
        print(f"Final position: {env.agent_pos}")
    
    return {
        "success": False,
        "total_reward": total_reward,
        "steps": step_count,
        "reason": "Goal not reached",
        "final_pos": tuple(env.agent_pos),
        "images": images
    }


def visualize_test_results(env, result, save_path=None):
    """可视化测试结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示初始状态
    env.reset()
    axes[0].imshow(env.render())
    axes[0].set_title(f"Initial State\nAgent: {env.agent_pos}, Goal: {env.unwrapped.goal_pos}")
    axes[0].axis('off')
    
    # 显示结果信息
    axes[1].axis('off')
    result_text = f"""
Test Results:
{'✅ SUCCESS' if result['success'] else '❌ FAILED'}

Total Reward: {result['total_reward']}
Total Steps: {result['steps']}

{'Reason: ' + result.get('reason', 'N/A') if not result['success'] else ''}
{'Final Position: ' + str(result.get('final_pos', 'N/A')) if not result['success'] else ''}
    """
    axes[1].text(0.1, 0.5, result_text, fontsize=14, verticalalignment='center',
                 fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例1: 简单路径测试
    print("="*60)
    print("Example 1: Simple path test")
    print("="*60)
    
    env = create_env_from_positions(
        size=8,
        agent_pos=(1, 1),
        agent_dir=0,  # 朝右
        goal_pos=(6, 6),
        render_mode='rgb_array'
    )
    
    # 测试路径: 右5步，下5步
    test_path = "R,R,R,R,R,D,D,D,D,D"
    result = test_path_with_reward(env, test_path, verbose=True, render_steps=True)
    
    visualize_test_results(env, result, save_path="test_result_1.png")
    
    # 示例2: 随机起点终点测试
    print("\n" + "="*60)
    print("Example 2: Random positions test")
    print("="*60)
    
    size = 10
    agent_pos = (random.randint(1, size-2), random.randint(1, size-2))
    goal_pos = (random.randint(1, size-2), random.randint(1, size-2))
    
    env2 = create_env_from_positions(
        size=size,
        agent_pos=agent_pos,
        agent_dir=random.randint(0, 3),
        goal_pos=goal_pos,
        render_mode='rgb_array'
    )
    
    # 生成随机路径
    random_path = ",".join([random.choice(['L', 'R', 'U', 'D']) for _ in range(10)])
    result2 = test_path_with_reward(env2, random_path, verbose=True)
    
    visualize_test_results(env2, result2, save_path="test_result_2.png")
    
    env.close()
    env2.close()