import torch.nn as nn

import torch 

class VGG(nn.Module):

    def __init__(self, num_classes=8, init_weights=True):
        super(VGG, self).__init__()
        self.classifier =  nn.Linear(256, 32, bias=True) 

    def forward(self, x):
        x = self.classifier(x)
        return x
    

model = VGG()
x  = torch.randn((1,2,256))
y = model(x)
print(model.classifier.weight)
print(model.classifier.weight.shape)
print(y)
torch.onnx.export(model, x, "model.onnx", opset_version=18)