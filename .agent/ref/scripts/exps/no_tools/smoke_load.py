#!/usr/bin/env python3
import argparse
import gc

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", required=True, type=int)
    parser.add_argument("--gpu-memory", required=True, type=float)
    args = parser.parse_args()

    model = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"image": 10},
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=8192,
        enforce_eager=True,
        seed=42,
    )
    outputs = model.generate(["Reply with OK."], SamplingParams(max_tokens=2, temperature=0))
    assert outputs and outputs[0].outputs, "smoke generation returned no output"
    print(f"SMOKE_OK model={args.model} tp={args.tp} output={outputs[0].outputs[0].text!r}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
