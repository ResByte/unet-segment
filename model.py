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
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
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
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        self.main = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 
                            kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, stride=1,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.upsample =  upsampling(in_channels, in_channels, 
                            bilinear=bilinear)
        
    def forward(self, x, y):
        x = self.upsample(x)
        # print(f"upsamples x : {x.shape} and y has {y.shape}")
        x = torch.cat([x, y], dim=1)
        # print(f"concat x : {x.shape}")
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
            outs = self.nb_filters*(2**(i)) 

            # create downsampling block for this ins-->outs
            down_conv = DownSample(ins, outs)

            self.down_convs.append(down_conv)
        
        # bottom 
        self.bottom = nn.Conv2d(outs, outs, kernel_size=3,stride=1, padding=1, bias=False)

        # start with previous depth and create upconv blocks
        for i in range(depth):
            ins = outs 
            outs = ins // 2

            # using bilinear up convolutions
            up_conv = UpSample(ins*2, outs, bilinear=True)
            self.up_convs.append(up_conv)
        
        #final layer
        self.out_conv = nn.Conv2d(outs, self.nb_classes, 
                            kernel_size=1, stride=1)
        
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        # TODO: add forward pass
        downs = [] 
        for down in self.down_convs:
            x, unpooled_x = down(x)
            print(f"downsampling with {x.shape}")
            downs.append(unpooled_x)

        x = self.bottom(x)

        for i,ups in enumerate(self.up_convs):
            print(f"Upsampling with {x.shape} and {downs[-i-1].shape}")
            x = ups(x,downs[-i-1])
            

        x = self.out_conv(x)

        return x   
        
if __name__=='__main__':
    
    x = Variable(torch.randn((1, 3, 512, 512)))
    model = Unet(depth=5,nb_classes=1)
    print(model)
    out = model(x)
    print(f"input with {x.shape} and output with {out.shape}")
    