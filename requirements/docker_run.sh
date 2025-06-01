docker run -it \
  --gpus all \
  --name my_container \
  -v /home/featurize/work/tool:/log \
  -w /app \
  --network host \
  67ea460be595 \
  /bin/bash

docker run -it \
  --gpus all \
  --name tool_server \
  -v /home/featurize/work/tool:/log \
  -w /app/OpenThinkIMG/ \
  --network host \
  tool_server:v0.2 