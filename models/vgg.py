# VGG model implemented by pytorch
# "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE SCALE IMAGE RECOGNITION"
# reference implementation https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch
import torch.nn as nn

# cfgs & _make_layers from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self,model='VGG16',num_classes=10,Batch_norm=True,init_weights=True):
        super(VGG,self).__init__()
        self.features=self._make_layers(cfgs[model],batch_norm=Batch_norm)
        # the original version of vgg net inpu is (224,244) so the output of the network is (512,7,7)
        # if your image does not have the size of (224,244) and the output should be different
        # therefore we can use nn.AdaptiveMaxPool2d to force the output changed to (512,7,7)
        self.avgpool=nn.AdaptiveMaxPool2d((7,7))
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 *7 , 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        out=self.features(x)
        # print(out.size())
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        # print(out.size())
        out=self.classifier(out)
        return out
    
    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
        

def test():
    vgg=VGG('VGG19')
    input=torch.randn(2,3,32,32)
    # print(input)
    output=vgg(input)
    print(output.size())

test()
