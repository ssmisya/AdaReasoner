from tool_server.tool_workers.tool_manager.base_manager import ToolManager

tool_manager = ToolManager(tools=[ "Draw2DPath", "Point"])

TOOL_PROMPTS = tool_manager.get_tool_prompt(prompt_type="one_tool_call")


GENERAL_PROMPT = """
# Task Description
You are a visual assistant skilled in solving visual reasoning problems, especially navigation tasks. Please help me construct data.
I am building a dataset that uses tool calls to answer Frozen Lake problems. Please help me rephrase the input conversation. Concretely, you need to add the appropriate thinking content and response content into it.
Please help me construct data for the following trajectory:
{TRAJECTORY_DEFINITION}
# Essential Information
I will provide you with some essential information:
### Original Problem Prompt:
{TASK_DEFINITION}

### Tool Call Prompt, please follow the format requirements here:
{TOOL_PROMPT}

# Notes
1. The tool inputs and outputs are provided. You need to add appropriate thinking content, i.e., content within <think></think> tags, response content, i.e., content within <response></response> tags, and connect them to form a complete dialogue.
2. The dialogue follows OpenAI format, for example: {{'role': 'user','content': [{{'type':'text','text': 'a question'}}, {{'type':"image",'image': 'images/0-0.jpg'}}]}} 
3. You can keep all image input or output as it was {{'type':"image",'image': 'images/0-0.jpg'}}
4. You don't need to generate a system_prompt, the dialogue roles are only user and assistant
"""

REASONING_REPHRASE_PROMPT = """

# Task Description
You are a visual assistant skilled in solving visual reasoning problems, especially navigation tasks. Please help me construct data.
I am building a dataset that uses tool calls to answer Frozen Lake problems. Please help me rephrase the reasoning statements within the input conversation.
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

PATH_FINDING_TRAJECTORY = """
The problem is to find the best path from the starting point (Elf) to the destination (Gift), avoiding dangerous ice holes (Ice Holes).
The trajectory to be constructed is:
1. First, call the Point tool to locate the Elf (starting point)
2. Then call the Point tool to locate the Gift (destination)
3. Then call the Point tool to locate the ice holes (Ice Holes)
4. Infer a path from Elf to Gift, avoiding any Ice Holes. Then call the Draw2DPath tool to visualize the trajectory of the path.
5. Based on the result from Draw2DPath, determine whether the path is correct. If it is correct, answer the question; if it is incorrect, revise the answer and repeat steps 4–5.

"""

PATH_VERIFY_TRAJECTORY = """
The problem is to determine whether a given control path from the starting point (Elf) is correct (can reach the destination (Gift) without passing through dangerous ice holes (Ice Holes)).
The trajectory to be constructed is:
1. First, call the Point tool to locate the Elf (starting point)
2. Then call the Draw2DPath tool to draw the trajectory of the given control path
3. Then judge whether this path is valid by looking at the image, and answer the question

"""

PATH_FINDING_TASK_INSTRUCTION = """
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


PATH_VERIFY_TASK_INSTRUCTION = """
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
3. Output if the action sequence is safe using "Yes" or "No". A safe action sequence should not include any unsafe actions.

Now please determine if the action sequence is safe for this given maze:

<TEST-IMAGE>

The action sequence is:

<ACTION-SEQ>
"""

PATH_FINDING_TASK_INSTRUCTION_SHORT = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.

Please generate action plan for the input maze image.
"""

PATH_VERIFY_TASK_INSTRUCTION_SHORT = """

You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{{Yes}} or \\boxed{{No}}.

The action sequence is:

<ACTION-SEQ>
"""



PATH_FINDING_FINAL_PROMPT = GENERAL_PROMPT.format(
    TRAJECTORY_DEFINITION=PATH_FINDING_TRAJECTORY,
    TASK_DEFINITION=PATH_FINDING_TASK_INSTRUCTION,
    TOOL_PROMPT=TOOL_PROMPTS,
)

PATH_VERIFY_FINAL_PROMPT = GENERAL_PROMPT.format(
    TRAJECTORY_DEFINITION=PATH_VERIFY_TRAJECTORY,
    TASK_DEFINITION=PATH_VERIFY_TASK_INSTRUCTION,
    TOOL_PROMPT=TOOL_PROMPTS,
)

REASONING_REPHRASE_FINAL_PROMPT = REASONING_REPHRASE_PROMPT.format(
    TRAJECTORY_DEFINITION=PATH_VERIFY_TRAJECTORY,
    TOOL_PROMPT=TOOL_PROMPTS,
)
PATH_FIDING_REASONING_REPHRASE_FINAL_PROMPT = REASONING_REPHRASE_PROMPT.format(
    TRAJECTORY_DEFINITION=PATH_FINDING_TRAJECTORY,
    TOOL_PROMPT=TOOL_PROMPTS,
)

PATH_FINDING_STAGE1 = TOOL_PROMPTS + PATH_FINDING_TASK_INSTRUCTION_SHORT