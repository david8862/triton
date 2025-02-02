import torch
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
image = torch.randn(1, 3, 224, 224)
resnet50_traced = torch.jit.trace(resnet50, image)
resnet50(image)
resnet50_traced.save('model.pt')
