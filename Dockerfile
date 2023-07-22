# FROM nvidia/cuda:11.4.3-base-ubuntu20.04
# FROM nvidia/cuda:11.3.1-base-ubuntu20.04
FROM nvidia/cuda:11.2.2-base-ubuntu20.04
WORKDIR /workspace
COPY requirements.txt .
RUN apt update
RUN apt -y install python3.8
RUN apt -y install python3-pip
RUN apt -y install python3.8-venv

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install python3.8-tk

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN source activate
RUN pip3 install -r /workspace/requirements.txt