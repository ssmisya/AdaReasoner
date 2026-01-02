# Evaluation for DeepEyes

We provide a evaluation demo for assess your model on V* benchmark with the bbox processing. 

## Evaluating Model
You can use the `eval_vstar.py` to evalate the model with the auto bbox processing. It is worth noting that we firstly deploy model using VLLM. If you want to use transformers to implement your model, you should modify the code and the evaluation process will be slow.

Here is a sample of the evaluation command：
```
python eval_vstar.py \
    --model_name MODEL_NAME \
    --api_key API_KEY \
    --api_url API_URL\
    --vstar_bench_path PATH_TO_VSTAR \
    --save_path PATH_TO_SAVE_DIR \
    --eval_model_name MODEL_NAME_VLLM \
    --num_workers NUM_WORKERS
```
`MODEL_NAME` is the name of saving, and the evaluation results will be saved at `PATH_TO_SAVE_DIR/MODEL_NAME`. `MODEL_NAME_VLLM` is the name of VLLM server. you can set `MODEL_NAME_VLLM` as None, and will be detected automatically. `API_URL` is the VLLM server port, such as 'http://10.39.19.140:8000/v1'.


## Score Calculate
We use the combination of ruled-based evaluation and llm-judge assessment to calculate score. You can use the following command to calculate your results:

```
python judge_result.py \
    --model_name MODEL_NAME \
    --api_key API_KEY \
    --api_url API_URL\
    --vstar_bench_path PATH_TO_VSTAR \
    --save_path PATH_TO_SAVE_DIR \
    --eval_model_name MODEL_NAME_VLLM \
    --num_workers NUM_WORKERS
```
We use Qwen2.5 72B deployed by VLLM as judge model, so `API_URL` is the address of judge model VLLM server. 


## Visualization
We also provide the code `watch_demo.ipynb` to visualize the result. You should modify the `root_path` to the V* bench path and `json_path` to the result jsonl path. Desides, you can modify `line_id` or `tosee_img` to change the case to be visualized.  

## Evaluate HRBench
The evaluation of HRBench is similar to that of V*.
