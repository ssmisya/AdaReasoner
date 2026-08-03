"""E5 fault-injection analysis: build the detect / recover / propagate matrix.

For each fault condition (vs a clean baseline run on the same fixed subset):
  - detected  : model shows an explicit reflection/re-call after the (early) faulted tool round
                (heuristic: a later <think> mentions doubt/retry/inconsistency OR it re-calls a tool
                 in a subsequent round after the injected round)
  - recovered : final answer is CORRECT despite the injected fault (score==1)
  - propagated: final answer is WRONG (score==0) AND no detection signal (silently used bad output)
  - accuracy delta vs baseline is also reported (the cleanest robustness proxy)

Usage: python e5_analyze.py <baseline_dir> <fault_dir1> [<fault_dir2> ...]
Each dir must contain vsp_ckpt.jsonl (+ vsp_result.jsonl for scores).
"""
import sys, json, os, re, glob

DOUBT = re.compile(r"(however|but |wait|re-?check|recheck|inconsistent|incorrect|does not|doesn'?t|"
                   r"seems? (wrong|off|odd)|mistake|error|retry|again|re-?examine|double-?check|"
                   r"contradict|not (correct|right|valid)|suspicious|unexpected|invalid)", re.I)

def load_jsonl(p):
    out=[]
    if not os.path.exists(p): return out
    for l in open(p):
        l=l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out

def load_scores(d):
    """idx(+task_type) -> score from vsp_result.jsonl compare_logs."""
    score={}
    for fp in glob.glob(os.path.join(d,"*result*.json*")):
        txt=open(fp).read()
        try: objs=[json.loads(txt)]
        except: objs=load_jsonl(fp)
        def walk(x):
            if isinstance(x,dict):
                if "idx" in x and "score" in x:
                    score[(x["idx"], x.get("task_type","?"))]=x["score"]
                for v in x.values(): walk(v)
            elif isinstance(x,list):
                for v in x: walk(v)
        for o in objs: walk(o)
    return score

def analyze_dir(d, inject_round=1):
    # ckpt filename varies by task (vsp_ckpt.jsonl / jigsaw_ckpt.jsonl / ...)
    ckpts = glob.glob(os.path.join(d, "*_ckpt.jsonl")) or glob.glob(os.path.join(d, "*ckpt*.jsonl"))
    recs = load_jsonl(ckpts[0]) if ckpts else []
    scores = load_scores(d)
    rows=[]
    for r in recs:
        res=r.get("results",{})
        idx=res.get("idx")
        inner=res.get("results",{})
        mrs=inner.get("model_response",[]) or []
        tcfg=inner.get("tool_cfg",[]) or []
        n_rounds=len(mrs)
        # detection: any <think> AFTER the injected round expresses doubt, OR a tool re-call after injected round
        detected=False
        for i in range(inject_round, len(mrs)):  # rounds after injection (0-idx: inject_round..)
            if isinstance(mrs[i],str) and DOUBT.search(mrs[i]):
                detected=True; break
        # tool re-call after injection round also counts as an active response
        recall_after = sum(1 for j in range(inject_round, len(tcfg)) if tcfg[j])
        if recall_after>0:
            detected=True
        # score for this idx (navigation & verify share idx prefix; take max over task_types present)
        sc=[v for (ix,tt),v in scores.items() if ix==idx]
        score = max(sc) if sc else None
        rows.append({"idx":idx,"n_rounds":n_rounds,"detected":detected,"score":score})
    return rows

def summarize(rows):
    n=len(rows)
    scored=[r for r in rows if r["score"] is not None]
    det=[r for r in rows if r["detected"]]
    correct=[r for r in scored if r["score"]==1]
    wrong=[r for r in scored if r["score"]==0]
    propagated=[r for r in wrong if not r["detected"]]   # wrong & no detection = silent propagation
    return {"n":n,"n_scored":len(scored),
            "detect_rate": round(len(det)/n,3) if n else None,
            "recover(acc)": round(len(correct)/len(scored),3) if scored else None,
            "propagate_rate": round(len(propagated)/len(scored),3) if scored else None,
            "avg_rounds": round(sum(r["n_rounds"] for r in rows)/n,2) if n else None}

def main():
    base=sys.argv[1]; faults=sys.argv[2:]
    print("=== E5 fault-injection analysis (early injection, VSP 100-subset) ===\n")
    base_rows=analyze_dir(base, inject_round=0)  # baseline no injection; detect meaningless
    bs=summarize(base_rows)
    print(f"{'condition':18s} {'n':>4} {'detect':>7} {'recover(acc)':>13} {'propagate':>10} {'avg_rounds':>10}")
    print(f"{'baseline':18s} {bs['n']:>4} {'-':>7} {str(bs['recover(acc)']):>13} {'-':>10} {bs['avg_rounds']:>10}")
    out={"baseline":bs, "faults":{}}
    for fd in faults:
        cond=os.path.basename(fd.rstrip('/')).replace("E5_","")
        rows=analyze_dir(fd, inject_round=1)  # early = round1 injected, look from round2 on
        s=summarize(rows)
        out["faults"][cond]=s
        acc_delta = (s['recover(acc)']-bs['recover(acc)']) if (s['recover(acc)'] is not None and bs['recover(acc)'] is not None) else None
        print(f"{cond:18s} {s['n']:>4} {str(s['detect_rate']):>7} {str(s['recover(acc)']):>13} {str(s['propagate_rate']):>10} {s['avg_rounds']:>10}   Δacc_vs_base={acc_delta}")
    outp=os.path.join(os.path.dirname(base.rstrip('/')),"E5_matrix.json")
    json.dump(out, open(outp,"w"), indent=2)
    print("\nsaved:", outp)

if __name__=="__main__":
    main()
