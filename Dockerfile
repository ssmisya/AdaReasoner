# === 基础镜像 ===
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# === 复制项目代码到容器中 ===
# 请确保构建命令的上下文路径（build context）是 tool-agent 的上一层目录
COPY ./OpenThinkIMG /app/OpenThinkIMG

ENV DEBIAN_FRONTEND=noninteractive
ENV https_proxy=http://172.16.0.13:5848
ENV HTTPS_PROXY=http://172.16.0.13:5848

# === 创建日志目录 ===
RUN mkdir -p /log

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv python3-dev \
    libgl1 libglib2.0-0 git


RUN git config --global http.sslVerify false

RUN cd /app/OpenThinkIMG/src
RUN rm -rf sam2 && rm -rf sam2 && rm -rf GroundingDINO
RUN git clone https://github.com/facebookresearch/sam2.git
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN pip install -e ./sam2/
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -e ./GroundingDINO/
RUN pip install -e /app/OpenThinkIMG

RUN pip install -r /app/OpenThinkIMG/apptainer/requirements.txt

RUN python -c "import easyocr; easyocr.Reader(['ch_sim','en'])"



# 在 Dockerfile 结尾前添加：
RUN python /app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
    --config tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml || exit 1

COPY ./weights/Molmo-7B-D-0924 /weights/Molmo-7B-D-0924
COPY ./weights/sam2-hiera-large /weights/sam2-hiera-large
COPY ./weights/groundingdino_swint_ogc.pth /weights/groundingdino_swint_ogc.pth
COPY ./weights/GroundingDINO_SwinT_OGC.py /weights/GroundingDINO_SwinT_OGC.py

# === 设置环境变量 ===
ENV PYTHONUNBUFFERED=1

# === 设置工作目录 ===
WORKDIR /app/OpenThinkIMG

# === 设置默认启动命令（等价于 %runscript）===
CMD ["python", "/app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/start_server_local.py", \
     "--config",  \
     "tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml"]

# === 镜像元信息（可选）===
LABEL maintainer="Mingyang Song <mysong23@m.fudan.edu.cn>"
LABEL version="1.0.0"
LABEL description="Tool Server Image"