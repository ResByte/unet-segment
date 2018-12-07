# unet-segment
Image segmentation model using U-Net style structure as described in [paper](https://arxiv.org/abs/1505.04597).

This repo provides an implementation of U-Net in `pytorch` with arbitrary large depth given sufficient image size. 

TODO:
- [ ] : Add dataset loader 
- [ ] : Training Script

## Dependencies 

- pytorch `>= 0.4.0` 

## Model

The api can be used as : 

```python
# create a dummy variable
x = Variable(torch.randn((1, 3, 512, 512)))

# load model and provide depth and number of output channels
model = Unet(depth=5,nb_classes=1)

# forward pass
out = model(x)

# To compare input and output sizes
print(f"input with {x.shape} and output with {out.shape}")
```

## Notebook 

Complete function notebook is available at [Colab Noteboo](./segmentation_unet.ipynb). Launch it with gpu instance. 


## Dataset 
- Paper provides resutls on : [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/)
- Other dataset that this model can work with : [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data)
- Recent popular [Places Challenge ICCV'17](http://placeschallenge.csail.mit.edu/) has good dataset : [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 

## References 
1. [Project page](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
2. [Tutorial on Medium](https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6) 


