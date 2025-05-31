docker run -it \
  --gpus all \
  --name my_container \
  -v /home/featurize/work/tool:/log \
  -w /app \
  --network host \
  67ea460be595 \
  /bin/bash