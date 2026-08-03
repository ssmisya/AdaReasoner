"""E4 reward-hacking self-check: join per-sample (tool-call count) x (correctness).
Design-intent framing (per user): the asymmetric reward is meant to elicit tool use ONLY when it helps.
So we want to show:
  (i)  on instances solved WITHOUT tools, accuracy is NOT lower than the tool-using regime
       -> i.e. the model skips tools on easy instances it already solves, no accuracy penalty
  (ii) on perception/planning-hard instances (here: navigation, and higher levels), tool-use rate is HIGH
       -> tools invoked when they help, not avoided when needed.

Inputs (produced by the E3 full VSP run):
  ckpt.jsonl   : per-idx tool_cfg (list of rounds, each a list of tool calls) -> count non-empty calls
  result file  : per-idx score (1/0). We recompute by re-reading compare_logs if present,
                 else fall back to reading the *_result.jsonl produced by the framework.
Usage: python e4_reward_hacking_selfcheck.py <ckpt.jsonl> <result_dir_or_file>
"""
import sys, json, os, glob

def load_jsonl(p):
    out=[]
    with open(p) as f:
        for l in f:
            l=l.strip()
            if l:
                try: out.append(json.loads(l))
                except: pass
    return out

def count_tool_calls(rec):
    """rec = one ckpt entry; results.results.tool_cfg is list[round] of list[call]."""
    inner = rec.get("results",{}).get("results",{})
    tcfg = inner.get("tool_cfg",[]) or []
    n=0
    for rnd in tcfg:
        if isinstance(rnd,list):
            for call in rnd:
                if isinstance(call,dict) and call.get("API_name"):
                    n+=1
    return n, inner

def main():
    ckpt = sys.argv[1]
    resloc = sys.argv[2]
    recs = load_jsonl(ckpt)
    print(f"ckpt records: {len(recs)}")

    # build idx -> tool_count, task_type, level
    info = {}
    for r in recs:
        idx = r.get("results",{}).get("idx")
        if not idx: continue
        n, inner = count_tool_calls(r)
        md = inner.get("meta_data",{})
        info[idx] = {"tool_calls": n,
                     "task_type": md.get("task_type") or ("navigation" if "nav" in idx.lower() else "verify" if "verify" in idx.lower() else "?"),
                     "level": md.get("level","?")}

    # find score per idx: search result files for compare_logs / per-sample score.
    # compare_logs entries carry (idx, task_type, level, score) authoritatively.
    # Key by (idx, task_type) since verify/navigation can share an idx.
    score = {}          # (idx, task_type) -> score
    logmeta = {}        # (idx, task_type) -> {task_type, level}
    files = []
    if os.path.isdir(resloc):
        files = glob.glob(os.path.join(resloc,"**","*.json*"), recursive=True)
    else:
        files = [resloc]
    for fp in files:
        try:
            txt = open(fp).read()
        except: continue
        # try full-json
        try:
            j = json.loads(txt)
            objs = [j]
        except:
            objs = load_jsonl(fp)
        for o in objs:
            # look for compare_logs anywhere
            def walk(x):
                if isinstance(x,dict):
                    if "idx" in x and "score" in x:
                        tt = x.get("task_type","?")
                        key = (x["idx"], tt)
                        score[key] = x["score"]
                        logmeta[key] = {"task_type": tt, "level": x.get("level","?")}
                    for v in x.values(): walk(v)
                elif isinstance(x,list):
                    for v in x: walk(v)
            walk(o)
    print(f"scored (idx,task_type) found: {len(score)}")

    # join: prefer result-file task_type/level; tool_calls from ckpt keyed by idx.
    # A ckpt idx maps to (possibly) multiple (idx,task_type) score keys — attach tool_calls to each.
    rows = []
    for (idx, tt), s in score.items():
        tc = info.get(idx,{}).get("tool_calls")
        lm = logmeta[(idx,tt)]
        rows.append({"idx": idx, "task_type": tt, "level": lm["level"],
                     "tool_calls": tc if tc is not None else 0,
                     "score": s})

    scored = [r for r in rows if r["score"] is not None]
    print(f"joined (with score): {len(scored)}")

    def rate(sub, cond):
        sub=[r for r in sub if cond(r)]
        return len(sub)

    # ---- (i) accuracy: tool-using vs no-tool subsets ----
    notool = [r for r in scored if r["tool_calls"]==0]
    withtool = [r for r in scored if r["tool_calls"]>0]
    def acc(sub):
        return (sum(r["score"] for r in sub)/len(sub)) if sub else None
    print("\n=== (i) accuracy by tool usage ===")
    print(f"  no-tool subset:   n={len(notool):4d}  acc={acc(notool)}")
    print(f"  tool-using subset:n={len(withtool):4d}  acc={acc(withtool)}")
    print(f"  overall:          n={len(scored):4d}  acc={acc(scored)}")

    # ---- (ii) tool-use rate by task hardness (task_type, level) ----
    print("\n=== (ii) tool-use rate by task_type ===")
    for tt in sorted(set(r["task_type"] for r in scored)):
        sub=[r for r in scored if r["task_type"]==tt]
        used=[r for r in sub if r["tool_calls"]>0]
        print(f"  {tt:12s}: n={len(sub):4d}  tool-use-rate={len(used)/len(sub):.3f}  avg_calls={sum(r['tool_calls'] for r in sub)/len(sub):.2f}  acc={acc(sub):.3f}")
    print("\n=== (ii) tool-use rate by level ===")
    for lv in sorted(set(str(r["level"]) for r in scored)):
        sub=[r for r in scored if str(r["level"])==lv]
        used=[r for r in sub if r["tool_calls"]>0]
        print(f"  level {lv}: n={len(sub):4d}  tool-use-rate={len(used)/len(sub):.3f}  avg_calls={sum(r['tool_calls'] for r in sub)/len(sub):.2f}  acc={acc(sub):.3f}")

    out = {"n_scored": len(scored),
           "notool": {"n": len(notool), "acc": acc(notool)},
           "withtool": {"n": len(withtool), "acc": acc(withtool)},
           "overall_acc": acc(scored),
           "by_task_type": {tt: {"n": len([r for r in scored if r["task_type"]==tt]),
                                 "tool_use_rate": len([r for r in scored if r["task_type"]==tt and r["tool_calls"]>0])/max(1,len([r for r in scored if r["task_type"]==tt])),
                                 "avg_calls": sum(r["tool_calls"] for r in scored if r["task_type"]==tt)/max(1,len([r for r in scored if r["task_type"]==tt])),
                                 "acc": acc([r for r in scored if r["task_type"]==tt])}
                            for tt in sorted(set(r["task_type"] for r in scored))}}
    outp = os.path.join(os.path.dirname(ckpt),"E4_selfcheck.json")
    json.dump(out, open(outp,"w"), indent=2)
    print("\nsaved:", outp)

if __name__=="__main__":
    main()
