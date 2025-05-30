docker run -it --rm \
  --name my_tool_container \
  --gpus all \
  -v /home/featurize/work/tool:/app/tool \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
  bash