from tool_server.tool_workers.tool_manager.base_manager import ToolManager

tool_manager = ToolManager()

TOOL_PROMPTS = tool_manager.get_tool_prompt(prompt_type="one_tool_call")


REASONING_REPHRASE_PROMPT = """

# Task Description
You are a visual assistant skilled in solving visual reasoning problems. Please help me construct data.
I am building a dataset that uses tool calls to answer Jigsaw problems. Please help me rephrase the reasoning statements within the input conversation.
I will provide you with the corresponding dialogue and indicate where the chain-of-thought information needs to be added. Please help me fill in the missing reasoning at those points to make the overall logic smooth and coherent.

Please help me construct data for the following trajectory:
{TRAJECTORY_DEFINITION}
# Essential Information
I will provide you with some essential information:

### Tool Call Prompt, please follow the format requirements here:
{TOOL_PROMPT}

# Notes
1. You need to rewrite all the highlighted think sections in the dialogue. The parts that require rewriting are formatted like this:
[**You need to implement this** Example: An example to show you how to reason.]
You should rewrite the example by considering its surrounding context.
2. Please return the output in the following format: <think> reasoning part 1 </think> <think> reasoning part 2 </think> ..., where each reasoning part is the chain-of-thought content you are asked to provide.
3. Please make sure that the number of <think> sections you return matches the number of <think> sections that were requested for rewriting.
4. You only need to return the rewritten reasoning steps. There's no need to include anything else.

"""

INSERT_VERIFY_TRAJECTORY = """
The problem is to find which candidate image correctly fills the missing part of the given picture.
The trajectory to be constructed is:

1. First, call the DetectBlackArea tool to locate the bounding box of the missing region in the picture.
2. Then, for each candidate option, call the InsertImage tool to insert the option image into the missing region until the correct answer is found.
3. Based on the results from InsertImage, determine whether the correct answer has been identified. If it has, provide the answer; if not, choose the most suitable option based on your own judgment and give it as the answer.
"""

SELF_THINK_TRAJECTORY = """
The problem is to find which candidate image correctly fills the missing part of the given picture.
The trajectory to be constructed is:
1. Analyze the spatial relationships in the picture and determine which candidate sub-image is most likely to be the correct answer.
2. Return the answer you consider to be correct.

"""

INSERT_VERIFY_PROMPT = REASONING_REPHRASE_PROMPT.format(
    TRAJECTORY_DEFINITION=INSERT_VERIFY_TRAJECTORY,
    TOOL_PROMPT=TOOL_PROMPTS,
)

SELF_THINK_PROMPT = REASONING_REPHRASE_PROMPT.format(
    TRAJECTORY_DEFINITION=SELF_THINK_TRAJECTORY,
    TOOL_PROMPT=TOOL_PROMPTS,
)