"""E3 tool-latency microbenchmark: expert-model tool (Point/Molmo) vs local tool (AStar).
Directly answers reviewer "CPS != cost": local operators are ~1e4x cheaper per call than expert-model tools.
"""
import os, sys, time, json, base64
from io import BytesIO
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/myangsong/AdaReasoner")

REPEAT = 20      # timed repeats per tool
WARMUP = 3

def pct(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*p/100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def stats(xs):
    return {"mean_ms": round(1000*sum(xs)/len(xs),3),
            "p50_ms": round(1000*pct(xs,50),3),
            "p90_ms": round(1000*pct(xs,90),3),
            "min_ms": round(1000*min(xs),3),
            "max_ms": round(1000*max(xs),3),
            "n": len(xs)}

results = {}

# ---- grab a real VSP image ----
import pandas as pd
pq = "/apdcephfs_cq11/share_1567347/share_info/myangsong/datasets/AdaEval-VSP/data/navigation_test-00000-of-00001.parquet"
df = pd.read_parquet(pq)
print("VSP nav columns:", list(df.columns), "rows:", len(df), flush=True)
# find an image column
img_col = None
for c in df.columns:
    v = df.iloc[0][c]
    if isinstance(v, dict) and ("bytes" in v or "path" in v):
        img_col = c; break
    if isinstance(v, (bytes, bytearray)):
        img_col = c; break
print("image column:", img_col, flush=True)
row = df.iloc[0]
raw = row[img_col]
if isinstance(raw, dict):
    raw = raw.get("bytes") or raw
img = Image.open(BytesIO(raw)).convert("RGB")
print("image size:", img.size, flush=True)
img_b64 = base64.b64encode(BytesIO(_bio:=BytesIO()).getvalue()).decode() if False else None
_b = BytesIO(); img.save(_b, format="PNG"); img_b64 = base64.b64encode(_b.getvalue()).decode()

# ---- local tool: AStar ----
from tool_server.tool_workers.offline_workers.astar import AStarWithPixelCoordinate
astar = AStarWithPixelCoordinate()
W, H = img.size
astar_params = {"start":[10,10], "goal":[W-10,H-10],
                "obstacles":[[100,100],[200,200],[300,300],[150,250],[250,150]],
                "cell_size":32}
for _ in range(WARMUP):
    astar.generate(dict(astar_params))
ts=[]
for _ in range(REPEAT):
    t=time.perf_counter(); r=astar.generate(dict(astar_params)); ts.append(time.perf_counter()-t)
results["AStar_local"] = stats(ts)
results["AStar_local"]["status"] = r.get("status")
print("AStar:", results["AStar_local"], flush=True)

# ---- expert-model tool: Point (Molmo) ----
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
MOLMO = "/home/myangsong/models/Molmo-7B-D-0924"
print("loading Molmo...", flush=True)
t0=time.perf_counter()
proc = AutoProcessor.from_pretrained(MOLMO, trust_remote_code=True, torch_dtype='auto', device_map='auto')
model = AutoModelForCausalLM.from_pretrained(MOLMO, trust_remote_code=True, torch_dtype='auto', device_map='auto')
model.eval()
print(f"Molmo loaded in {time.perf_counter()-t0:.1f}s, mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

@torch.inference_mode()
def point_call(desc="the goal marker"):
    inputs = proc.process(images=[img], text=f"Point to the {desc} in the scene.")
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model.generate_from_batch(inputs,
              GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
              tokenizer=proc.tokenizer)
    gen = out[0, inputs['input_ids'].size(1):]
    return proc.tokenizer.decode(gen, skip_special_tokens=True)

for _ in range(WARMUP):
    point_call()
torch.cuda.synchronize()
ts=[]
for _ in range(REPEAT):
    torch.cuda.synchronize(); t=time.perf_counter()
    resp=point_call(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
results["Point_Molmo_expert"] = stats(ts)
results["Point_Molmo_expert"]["sample_response"] = resp[:80]
print("Point:", results["Point_Molmo_expert"], flush=True)

# ---- ratio ----
ratio = results["Point_Molmo_expert"]["mean_ms"] / results["AStar_local"]["mean_ms"]
results["expert_vs_local_ratio"] = round(ratio, 1)
print(f"\n=== Point(expert) / AStar(local) = {ratio:.0f}x ===", flush=True)

out_path = "/home/myangsong/AdaReasoner/rebuttal_exps/E3_tool_latency.json"
with open(out_path,"w") as f: json.dump(results, f, indent=2)
print("saved:", out_path, flush=True)
