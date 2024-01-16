import numpy as np
import tritonclient.http as httpclient
import torch
from PIL import Image


if __name__ == '__main__':
    #triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    triton_client = httpclient.InferenceServerClient(url='192.168.50.102:8000')
    image = Image.open('./cat.jpg')

    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('image_input', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('scores', binary_data=False, class_count=3))  # class_count 表示 topN 分类
    # outputs.append(httpclient.InferRequestedOutput('scores', binary_data=False))

    results = triton_client.infer('resnet50_onnx', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('scores')
    print(output_data0.shape)
    print(output_data0)
