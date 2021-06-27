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
GoogleNet, VGG (2014) : deeper networks
<br><br><br>

## VGGNet (2014)
Small filters, Deeper networks (8 layers → 16~19 layers)<br>
<img width="190" alt="스크린샷 2021-06-27 오후 6 05 53" src="https://user-images.githubusercontent.com/67621291/123538976-52c7c700-d772-11eb-988d-acde4b81e7c9.png"><br>
`Q)` Why use smalelr filters? (3x3 conv)<br>
`A)` Stack of three 3x3 conv (stride 1) layers has same **effective receptive field** as one 7x7 conv layer<br>
But deeper, more non-linearities<br>And fewer parameters: 3 * (3^2 * C^2) vs 7^2 * C^2 for C channels per layer<br><br>
Heavy memory <br>
**Most memory is in early CONV / Most params are in late FC** <br><br>
Details:
- Similar training procedure as Krizhevsky 2012
- No Local Response Normalisation (LRN)
- Use VGG16 or VGG19 (VGG19 only slightly better, more memory)
- FC7 features generalize well to other tasks
<br><br><br>

## GoogLeNet (2014)
Deeper networks, with computational efficiency
- 22 layers
- Efficient "Inception" module
- No FC layers
- Only 5 million parameters (12x less than AlexNet)

> Inception module : <br>
> design a good local network topology(network within network) and then stack these modules on top of each other<br>
> <img width="300" alt="스크린샷 2021-06-27 오후 6 12 21" src="https://user-images.githubusercontent.com/67621291/123539152-39734a80-d773-11eb-98df-2e68533d6d0b.png"><br>
> Apply parallel filter operations on the input from previous later: <br>
>> Multiple receptive field size for convolutoin (1x1, 3x3, 5x5)<br>
>> Pooling operation (3x3)<br>
> <br>
> Concatenate all filter outputs together depth-wise <br>
> 
> **PROBLEM:** very expensive compute <br>
> Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer<br>
> <img width="700" alt="스크린샷 2021-06-27 오후 6 13 33" src="https://user-images.githubusercontent.com/67621291/123539188-64f63500-d773-11eb-94e8-8e1fc7df133d.png">
<br><br>
> ➡️ **SOLUTION: bottleneck** layers that use 1x1 convolutions to reduce feature depth

<img width="364" alt="스크린샷 2021-06-27 오후 6 18 41" src="https://user-images.githubusercontent.com/67621291/123539319-1ac18380-d774-11eb-8c01-ec1e180ca56f.png">
Bottleneck can reduce operations<br><br><br>

<img width="679" alt="스크린샷 2021-06-27 오후 6 19 54" src="https://user-images.githubusercontent.com/67621291/123539350-47759b00-d774-11eb-9c3e-7d1eb2cd27d6.png">

Stem Network: Conv-Pool-2x Conv-Pool → Stacked Inception Modules → Classifier output (removed expensive FC layers)<br>
+) Auxiliary classification outputs to inject additional gradient at lower layers (AvgPool-1x1 Conv-FC-FC-Softmax)<br>
➡️ 22 total layers with weights (including each parallel layer in an Inception module)
<br><br><br>

## ResNet
Very deep networks using residual connections<br>
- 152 layer model for Imagenet
- 3.57% top 5 error
- Swept all classification and detection competitions in ILSVRC'15 and COCO'15
<br>

When we continue stacking deeper layers on a "plain" convolutional neural network, error is getting worse. <br>
<img width="500" alt="스크린샷 2021-06-27 오후 6 21 41" src="https://user-images.githubusercontent.com/67621291/123539401-873c8280-d774-11eb-9312-154aa49df430.png">
<br>
Network more deeper → training error will be lower (because of overfitting) HOWEVER, in this graph, deeper network shows higher training error<br>
`A)` The deeper model should be able to perform at least as well as the shallower model. <br>
A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping. <br>
**Solution:** Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping <br>
<img width="499" alt="스크린샷 2021-06-27 오후 6 23 02" src="https://user-images.githubusercontent.com/67621291/123539432-b7842100-d774-11eb-8982-921126dec058.png">

<br><br>
**Full ResNet architecture:**
- Stack residual blocks
- Every residual block has two 3x3 conv layes
- Periodically, double # of filters and downsample spatially using stride 2 (/2 in each dimension)
- Additional conv layer at the beginning
- No FC layers at the end (only FC 1000 to output classes)
<img width="352" alt="스크린샷 2021-06-27 오후 6 26 24" src="https://user-images.githubusercontent.com/67621291/123539533-38dbb380-d775-11eb-8b96-ae4130ff890f.png">

Total depths of 34, 50, 101, or 152 layers for ImageNet<br>
For deeper networks (ResNet-50+), use "bottleneck" layer to improve efficiency (similar to GoogLeNet)
<img width="349" alt="스크린샷 2021-06-27 오후 6 27 58" src="https://user-images.githubusercontent.com/67621291/123539578-69235200-d775-11eb-95da-42a4418c136d.png">

<br>

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
<img width="500" alt="스크린샷 2021-06-27 오후 6 29 57" src="https://user-images.githubusercontent.com/67621291/123539632-af78b100-d775-11eb-909a-a1a46f1e65a9.png">

Size of circle means memory 

<br><br>
## Other architectures to know
### Network in Network (NiN) (2014)
- Mlpconv layer with "micronetwork" within each conv layer to compute more abstract features for local patches
- Micronetwork uses multilayer perceptron
- Precursor to GoogLeNet and RestNet "bottleneck" layers
- Philosophical inspiration for GoogLeNet

<img width="420" alt="스크린샷 2021-06-27 오후 6 31 14" src="https://user-images.githubusercontent.com/67621291/123539699-dcc55f00-d775-11eb-8a26-525b04d6818d.png"><br><br>

### Identity Mappings in Deep Residual Networks (2016)
- Improved ResNet block design from creators of ResNet
- Creates a more direct path for propagating information throughout network (moves activation to residual mapping pathway)
- Gives better performance

<img width="163" alt="스크린샷 2021-06-27 오후 6 32 05" src="https://user-images.githubusercontent.com/67621291/123539726-fbc3f100-d775-11eb-8924-36915bcc9179.png"><br><br>

### Wide Residual Networks (2016)
- Argues that residuals are the important factor, not depth 
- User wider residual blocks (Fxk filters instead of F filters in each layer)
- 50-layer ResNet outperforms 152-layer original ResNet
- Increasing width instead of deapth more compuatationally efficient (parallelizable)

<img width="346" alt="스크린샷 2021-06-27 오후 6 32 32" src="https://user-images.githubusercontent.com/67621291/123539740-0c746700-d776-11eb-895f-1b43acf1d16b.png"><br><br>

Improving ResNets
### Aggregated Residual Transformations for Deep Neural Networks (ResNeXt) (2016)
- Also from creators of ResNet
- Increases width of residual block through multiple parallel pathways ("cardinality")
- Parallel pathways similar in spirit to Inception module

<img width="346" alt="스크린샷 2021-06-27 오후 6 33 09" src="https://user-images.githubusercontent.com/67621291/123539751-20b86400-d776-11eb-8051-117aca45f37f.png"><br>

### Deep Networks with Stochastic Depth (2016)
- Motivation: reduce vanishing gradients and training time through short networks during training 
- Randomly drop a subset of layers during each training pass
- Bypass with identity fuction
- Use full deep network at test time

<img width="130" alt="스크린샷 2021-06-27 오후 6 33 41" src="https://user-images.githubusercontent.com/67621291/123539763-33cb3400-d776-11eb-873c-365e35594fca.png"><br><br>
 
Beyond ResNets
### FractalNet: Ultra-Deep Neural Networks without Residuals (2017)
- Argues that key is transitioning effectively from shallow to deep and residual representations are not necessary
- Fractal architecture with both shallow and deep paths to output 
- Trained with dropping out sub-paths 
- Full network at test time

<img width="389" alt="스크린샷 2021-06-27 오후 6 34 12" src="https://user-images.githubusercontent.com/67621291/123539768-46456d80-d776-11eb-8b3e-15ccce2583e6.png"><br>

### Densely Connected Convolutional Networks (2017)
- Dense blocks where each layer is connected to every other layer in feedforward fashion
- Alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse

<img width="262" alt="스크린샷 2021-06-27 오후 6 34 48" src="https://user-images.githubusercontent.com/67621291/123539787-5b220100-d776-11eb-804e-81a4c16a2128.png"><img width="102" alt="스크린샷 2021-06-27 오후 6 35 01" src="https://user-images.githubusercontent.com/67621291/123539791-637a3c00-d776-11eb-9ffa-a291ba2547d1.png"><br><br>

Efficient networks
### SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5Mb Model Size (2017)
- Fire modules consisting of a 'squeeze' layer with 1x1 filters feeding an 'expand' layer with 1x1 and 3x3 filters
- AlexNet level accuracy on ImageNet with 50x fewer parameters
- Can compress to 510x smaller than AlexNet (0.5Mb)

<img width="386" alt="스크린샷 2021-06-27 오후 6 35 24" src="https://user-images.githubusercontent.com/67621291/123539803-70972b00-d776-11eb-8b14-4a4261bf3e36.png">
