docker build -t develop
docker run -it --ipc=host --gpus all --name test develop bash