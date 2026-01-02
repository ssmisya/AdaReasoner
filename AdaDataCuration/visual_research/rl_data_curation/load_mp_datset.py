from tool_server.utils.utils import setup_proxy
setup_proxy()
from datasets import load_dataset

vs_dataset = load_dataset("ThinkMorph/Visual_Search")

print(vs_dataset)

print(f"Lengh: {len(vs_dataset['train'])}",split="train")