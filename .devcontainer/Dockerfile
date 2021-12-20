# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYTORCH_VERSION=21.11

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"


RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info

COPY . /workspace

RUN \
     cd /workspace && \
# Installationn
    pip install "Pillow>=8.2, !=8.3.0" "cryptography>=3.4" "py>=1.10" --no-cache-dir --upgrade-strategy only-if-needed && \
    pip install -r ./requirements/extra.txt --no-cache-dir --upgrade-strategy only-if-needed && \
    pip install -r ./requirements/example.txt --no-cache-dir --upgrade-strategy only-if-needed &&\
    pip install jupyterlab[all] -U && \
    pip list

    RUN pip install lightning-grid -U && \
        pip install "py>=1.10" "protobuf>=3.15.6" --upgrade-strategy only-if-needed

    ENV PYTHONPATH="/workspace"

RUN \
    ls -a  && \
    pip install -r ./requirements/example.txt --no-cache-dir --upgrade-strategy only-if-needed

RUN \
    python --version && \
    pip --version && \
    pip list | grep torch

    CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
