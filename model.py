import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from torch.autograd import Variable
print(torch.__version__)

def upsampling(in_channels, out_channels, bilinear=False):
    if bilinear :
        return nn.Upsample(mode='bilinear',
                    scale_factor=2,
                    align_corners=True)
    else:
        return nn.ConvTranspose2d(in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2)




class DownSample(nn.Module):
    """
    Downsampling block in the U-Net structure.
    This consists of :
        - Conv3x3 + relu
        - Conv3x3 + relu
        - maxpool2x2
    

    """
    def __init__(self, in_channels=3, out_channels=64):
        super(DownSample, self).__init__()
        
        self.main = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, stride=1, padding=1),
                        nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.main(x)
        return self.pool(x), x




class UpSample(nn.Module):
    """
    Upsampling module of the U-Net Structure
    This consists of :
        - conv 3x3 + relu
        - conv 3x3 + relu
        - upsample/upconv 2x2
    Params:
        - Cropped layer from previous
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpSample, self).__init__()
        self.main = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 
                            kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, stride=1,padding=1),
                        nn.ReLU(),
                        upsampling(out_channels, out_channels, 
                            bilinear=bilinear)

        )
        
    def forward(self, x,y):
        x = torch.cat((x, y), dim=0)
        print(x.shape)
        return self.main(x)

if __name__=='__main__':
    up_sample = UpSample(3, 32, bilinear=True)
    x = Variable(torch.randn((1, 3, 320, 320)))

    out = up_sample(x, x)
    print(out.shape)

    down_sample = DownSample(in_channels=3,out_channels=64)
    out_pooled, out = down_sample(x)
    print(out.shape, out_pooled.shape)