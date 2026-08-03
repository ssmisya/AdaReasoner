"""离线 LLM judge (rebuttal E6): 复用框架 llm_eval_webmmu 的一致性prompt, 用本地vLLM批推理给
GUIChat/WebMMU的开放式QA打0/1分。不起server(避免vLLM orphan), 一次性批处理。
用法: python e6_offline_judge.py <judge_model_path> <result.jsonl> [<result.jsonl> ...]
"""
import sys, json, os

# ── 复用框架的judge prompt(不自造标准) ──
CHAT_TEMPLATE = """
You are an expert evaluator. Your goal is to determine if a [Model Answer] correctly and factually answers a [Question] when compared against a [Standard Answer].

**Core Evaluation Principle:**
The [Model Answer] is considered consistent if it contains the **essential key information** present in the [Standard Answer]. The [Model Answer] is allowed to be much more verbose, conversational, and include additional correct context or explanations. Your primary task is to **verify the presence of the core facts**, not to penalize extra information. **If a question asks for specific formatting like coordinates or tables, but the model identifies the correct core element textually, it should still be considered consistent.**

- **Consistent (Judgement: 1):** The [Model Answer] successfully identifies the main point or action from the [Standard Answer]. For example, if the standard answer is to "click button A", the model answer is consistent if it mentions clicking or interacting with "button A", even if it's surrounded by other text.
- **Inconsistent (Judgement: 0):** The [Model Answer] fails to mention the key information, provides contradictory information, or hallucinates a different solution.

**Output Format:**
Just output `Judgement: 1` or `Judgement: 0`. Do not output anything else.
"""

def full_prompt(pred, gold, q):
    return (CHAT_TEMPLATE + "\n\n" +
            f"[Question]: {q}\n[Standard Answer]: {gold}\n[Model_answer] : {pred}\nJudgement:")

def parse_verdict(text):
    t = text.strip()
    if 'Judgement:' in t:
        t = t.split('Judgement:')[-1].strip()
    if '1' in t and '0' not in t.split('1')[0]:  # first digit is 1
        return 1
    return 1 if (t.strip().startswith('1')) else 0

def main():
    judge_model = sys.argv[1]
    # 可选: 第2个参数是 tp:N 指定tensor_parallel (72B需要); 其余为待judge文件
    args = sys.argv[2:]
    tp = 1
    if args and args[0].startswith("tp:"):
        tp = int(args[0].split(":")[1]); args = args[1:]
    files = args
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    llm = LLM(model=judge_model, tensor_parallel_size=tp,
              gpu_memory_utilization=0.90, max_model_len=8192, enforce_eager=True)
    tok = AutoTokenizer.from_pretrained(judge_model)
    sp = SamplingParams(temperature=0.0, max_tokens=12)

    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        r = recs[0]
        cl = r.get('compare_logs', [])
        # 构造chat prompts
        prompts = []
        for item in cl:
            msgs = [{"role": "user", "content": full_prompt(
                str(item.get('pred')), str(item.get('gold')), str(item.get('question')))}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)
        scores = [parse_verdict(o.outputs[0].text) for o in outs]
        acc = sum(scores) / len(scores) if scores else 0.0
        # 写回
        for item, s in zip(cl, scores):
            item['llm_score'] = s
        r['llm_judge_acc'] = acc
        r['llm_judge_model'] = os.path.basename(judge_model)
        # 按category分组(WebMMU需要: Functional=论文Act.)
        cat_acc = {}
        cats = {}
        for item, s in zip(cl, scores):
            c = item.get('category')
            if c is not None:
                cats.setdefault(c, []).append(s)
        for c, v in cats.items():
            cat_acc[c] = sum(v) / len(v)
        if cat_acc:
            r['llm_judge_cat_acc'] = cat_acc
        outp = f.replace('.jsonl', '_judged.jsonl')
        with open(outp, 'w') as fo:
            fo.write(json.dumps(r) + "\n")
        catstr = ("  cat=" + json.dumps({k: round(v, 4) for k, v in cat_acc.items()})) if cat_acc else ""
        print(f"JUDGE {os.path.basename(os.path.dirname(f))}: llm_acc={acc:.4f} (n={len(scores)}){catstr} -> {outp}")

if __name__ == "__main__":
    main()
