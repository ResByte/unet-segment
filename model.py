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
        # backbone of downsampling 
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
                        nn.Conv2d(in_channels*2, out_channels, 
                            kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, stride=1,padding=1),
                        nn.ReLU(),
                        upsampling(out_channels, out_channels, 
                            bilinear=bilinear)

        )
        
    def forward(self, x,y):
        x = torch.cat((x, y), dim=1)
        return self.main(x)


class Unet(nn.Module):
    """
    Overall unet structure 
    Ex: for a depth of 1
    conv3x3 ->conv3x3-->crop_and_add-->conv3x3-->conv3x3-->conv1x1--> out
                 |                      / \
                \ /                      |
            max_pool2x2              upsample2x2
                 |                      / \
                \ /                      |
               conv3x3        -->     conv3x3
    """
    def __init__(self,
        in_channels=3, 
        nb_filters = 64,
        depth = 3,
        nb_classes=10,
        upsample_mode='bilinear'):
        
        super(Unet,self).__init__()
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.depth = depth
        self.nb_classes = nb_classes
        self.upsample_mode = upsample_mode
        self.down_convs = []
        self.up_convs = []

        for i in range(depth):
            if i ==0:
                ins = self.in_channels
            else:
                ins = outs
            # increase the number of filters as  2^depth
            outs = self.nb_filters*(2**(i+1)) 

            # create downsampling block for this ins-->outs
            down_conv = DownSample(ins, outs)

            self.down_convs.append(down_conv)
        
        # start with previous depth and create upconv blocks
        for i in range(depth):
            ins = outs 
            outs = ins // 2

            # using bilinear up convolutions
            up_conv = UpSample(ins, outs, bilinear=True)
            self.up_convs.append(up_conv)
        
        #final layer
        self.out_conv = nn.Conv2d(outs, self.nb_classes, 
                            kernel_size=1, stride=1)
        
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
       # TODO: add forward pass

       # process downsampling 

       # process cropping of each intermediate downsampling layer 

       # append crops and process upsampling 
       # return final layers 
       
       return  
        
if __name__=='__main__':
    
    x = Variable(torch.randn((1, 3, 320, 320)))
    print("downsampling")
    down_sample = DownSample(in_channels=3,out_channels=64)
    out_pooled, out = down_sample(x)
    print(out.shape, out_pooled.shape)
    print("upsampling")
    up_sample = UpSample(64, 3, bilinear=True)
    out = up_sample(out_pooled, out_pooled)
    print(out.shape, x.shape)

    