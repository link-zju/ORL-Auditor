# FROM nvidia/cuda:11.4.3-base-ubuntu20.04
# FROM nvidia/cuda:11.3.1-base-ubuntu20.04
FROM nvidia/cuda:11.2.2-base-ubuntu20.04
WORKDIR /workspace
COPY requirements.txt .
RUN apt update
RUN apt -y install python3.8
RUN apt -y install python3-pip
RUN python3 -m pip install --upgrade pip
RUN apt -y install python3.8-venv

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt -y install python3.8-tk

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN source activate
RUN pip3 install  torch==1.12.1+cu113  torchvision==0.13.1+cu113  torchaudio==0.12.1  --index-url https://download.pytorch.org/whl/cu113  
RUN pip3 install -r /workspace/requirements.txt


