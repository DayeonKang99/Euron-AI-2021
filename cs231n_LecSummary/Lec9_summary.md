# CNN Architectures
## AlexNet (2012)
Input: 227x227x3 images → <br>CONV1 layer: 96 11x11 filters applied at stride 4 <br>
→ output: (227 -11)/4 + 1 = 55.  55x55x96<br>
parameters: (11 * 11 * 3) * 96 = 35K<br><br>
POOL1 layer: 3x3 filters applied at stride 2<br>
→ output: (55 - 2) / 2 + 1 = 27.  27x27x96<br>
parameters: 0
<br><br>
Details/Retrospectives:<br>
- first use of ReLU
- used Norm layers (not common anymore)
- heavy data augmentation
- dropout 0.5
- batch size 128
- SGD Momentum 0.9
- Learning rate 1e-2, reduced by 10 manually when val accuracy plateaus
- L2 weight decay 5e-4
- 7 CNN ensemble: 18.2% to 15.4%

<br><br>
ZFNet(2013) : change stride size, # of filters → improve the error rate but basically the same idea<br>
GoogleNet, VGG (2014) : deeper networks<br><br>

## VGGNet (2014)
Small filters, Deeper networks (8 layers → 16~19 layers)<br>
<img width="190" alt="스크린샷 2021-06-27 오후 6 05 53" src="https://user-images.githubusercontent.com/67621291/123538976-52c7c700-d772-11eb-988d-acde4b81e7c9.png"><br>
`Q)` Why use smalelr filters? (3x3 conv)<br>
`A)` Stack of three 3x3 conv (stride 1) layers has same **effective receptive field** as one 7x7 conv layer<br>
But deeper, more non-linearities<br>And fewer parameters: 3 * (3^2 * C^2) vs 7^2 * C^2 for C channels per layer<br><br>
Heavy memory <br>
**Most memory is in early CONV / Most params are in late FC** <br><br>
Details:
- 넣기
<br><br>

## GoogLeNet (2014)
Deeper networks, with computational efficiency
- 22 layers
- Efficient "Inception" module
- No FC layers
- Only 5 million parameters (12x less than AlexNet)

> Inception module : <br>
> design a good local network topology(network within network) and then stack these modules on top of each other<br>
> 사진 넣기
> Apply parallel filter operations on the input from previous later: <br>
>> Multiple receptive field size for convolutoin (1x1, 3x3, 5x5)<br>
>> Pooling operation (3x3)
> Concatenate all filter outputs together depth-wise <br>
> **PROBLEM:** very expensive compute <br>
> Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer<br>
> 사진 넣기<br>
> ➡️ **SOLUTION: bottleneck** layers that use 1x1 convolutions to reduce feature depth
bottlenect layer 사진
Bottleneck can reduce operations<br><br>
**Full GoogLeNet architecture**<br>
사진
Stem Network: Conv-Pool-2x Conv-Pool → Stacked Inception Modules → Classifier output (removed expensive FC layers)<br>
+ Auxiliary classification outputs to inject additional gradient at lower layers (AvgPool-1x1 Conv-FC-FC-Softmax)<br>
➡️ 22 total layers with weights (including each parallel layer in an Inception module)<br><br>

## ResNet
Very deep networks using residual connections<br>
- 152 layer model for Imagenet
- 3.57% top 5 error
- Swept all classification and detection competitions in ILSVRC'15 and COCO'15

When we continue stacking deeper layers on a "plain" convolutional neural network, error is getting worse. <br>
그래프 삽입 
Network more deeper → training error will be lower (because of overfitting) HOWEVER, in this graph, deeper network shows higher training error<br>
`A)` The deeper model should be able to perform at least as well as the shallower model. <br>
A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping. <br>
**Solution:** Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping <br>
residual block 사진 
<br><br>
Full ResNet architecture:
- Stack residual blocks
- Every residual block has two 3x3 conv layes
- Periodically, double # of filters and downsample spatially using stride 2 (/2 in each dimension)
- Additional conv layer at the beginning
- No FC layers at the end (only FC 1000 to output classes)
사진
Total depths of 34, 50, 101, or 152 layers for ImageNet<br>
For deeper networks (ResNet-50+), use "bottleneck" layer to improve efficiency (similar to GoogLeNet)
사진
<br><br>
Training ResNet in practice:
- Batch Normalization after every CONV layer
- Xavier/2 initialization from He et al.
- SGD + Momentum (0.9)
- Learning rate: 0.1, divided by 10 when validation error plateaus 
- Mini-batch size 256 
- Weight decay of 1e-5
- No dropout used 
<br>

Experimental Results 
- Able to train very deep networks without degrading 
- Deeper networks now achieve lowing training error as expected

<br><br><br>
그래프  
Size of circle == memory 

<br><br>
## Other architectures to know
### Network in Network (NiN) (2014)
- Mlpconv layer with "micronetwork" within each conv layer to compute more abstract features for local patches
- Micronetwork uses multilayer perceptron
- Precursor to GoogLeNet and RestNet "bottleneck" layers
- Philosophical inspiration for GoogLeNet
사진
<br><br>

### Identity Mappings in Deep Residual Networks (2016)
- Improved ResNet block design from creators of ResNet
- Creates a more direct path for propagating information throughout network (moves activation to residual mapping pathway)
- Gives better performance
사진
<br><br>

### Wide Residual Networks (2016)
- Argues that residuals are the important factor, not depth 
- User wider residual blocks (Fxk filters instead of F filters in each layer)
- 50-layer ResNet outperforms 152-layer original ResNet
- Increasing width instead of deapth more compuatationally efficient (parallelizable)
사진
<br><br>

Improving ResNets
### Aggregated Residual Transformations for Deep Neural Networks (ResNeXt) (2016)
- Also from creators of ResNet
- Increases width of residual block through multiple parallel pathways ("cardinality")
- Parallel pathways similar in spirit to Inception module

<br>
### Deep Networks with Stochastic Depth (2016)
- Motivation: reduce vanishing gradients and training time through short networks during training 
- Randomly drop a subset of layers during each training pass
- Bypass with identity fuction
- Use full deep network at test time 
사진
<br><br>
 
Beyond ResNets
### FractalNet: Ultra-Deep Neural Networks without Residuals (2017)
- Argues that key is transitioning effectively from shallow to deep and residual representations are not necessary
- Fractal architecture with both shallow and deep paths to output 
- Trained with dropping out sub-paths 
- Full network at test time
사진

<br>
### Densely Connected Convolutional Networks (2017)
- Dense blocks where each layer is connected to every other layer in feedforward fashion
- Alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse
사진
<br><br>

Efficient networks
### SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5Mb Model Size (2017)
- Fire modules consisting of a 'squeeze' layer with 1x1 filters feeding an 'expand' layer with 1x1 and 3x3 filters
- AlexNet level accuracy on ImageNet with 50x fewer parameters
- Can compress to 510x smaller than AlexNet (0.5Mb)
사진
