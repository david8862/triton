import torch
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
image = torch.randn(1, 3, 224, 224)

torch.onnx.export(resnet50, image, 'model.onnx', verbose=False, opset_version=12, input_names=['image_input'], output_names=['scores'])
                  # dynamic_axes={"image_input": {0: "batch_size"}, "scores": {0: "batch_size"}})


#resnet50_traced = torch.jit.trace(resnet50, image)
#resnet50(image)
#resnet50_traced.save('model.pt')
