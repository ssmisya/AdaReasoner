from openai import OpenAI
import requests
import random
import re
import os

from math_verify import parse, verify


def compute_score(data_sources, solution_strs, ground_truths, tool_rewards):
    # 每个的形状都是batch_size*n，包括模型的回答solution_strs
    # 但是solution_strs可能会包含多轮，但是由于是文本，所以没有明确的分割标识
    print(f"啊啊啊data_sources的形状是: {len(data_sources)}")
    print(f"啊啊啊solution_strs的形状是: {len(solution_strs)}")
    print(f"啊啊啊ground_truths的形状是: {len(ground_truths)}")
    print(f"啊啊啊tool_rewards的形状是: {len(tool_rewards)}")
    print("==========================================")
    print(f"啊啊啊data_sources[0]是: {data_sources[0]}")
    print(f"啊啊啊solution_strs[0]是: {solution_strs[0]}")
    print(f"啊啊啊ground_truths[0]是: {ground_truths[0]}")
    print(f"啊啊啊tool_rewards[0]是: {tool_rewards[0]}")

    return [1] * len(data_sources)
