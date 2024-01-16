#!/bin/bash
# from https://github.com/triton-inference-server/server/blob/main/docs/examples/fetch_models.sh
set -ex

# ONNX densenet
mkdir -p 1
wget -O 1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
