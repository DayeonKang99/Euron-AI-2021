# Convolutional Neural Networks
### History
1957: Frank Rosenblatt developed the Mark I Perceptron machine
> Mark I Perceptron machine was the first implementation of the perceptron algorithm. \
> wx + b → update rule for w\
> output : 1/0

1960: Widrow and Hoff developed Adaline and Madaline
> Adaline and Madaline start to stack linear layers into multilayer perceptron networks

1986: Rumelhart introduced back-propagation\
2006: Hinton and Salakhutdinov. Reinvigorated research in Deep Learning \
2012: First strong results using for speech recognition by Hinton's lab\
2012: Imagenet classification with deep convolutional neural network by Alex Krizhevsky (ConvNet is first introduced)<br><br>
**What gave rise to ConvNet specifically** <br>
Hubel & Wiesel: try to understand how neurons in the visual cortex work → experiments
> 고양이의 뇌에 전극을 꽂고 다른 시각적 자극을 줌
>  자극에 반응하는 뉴런 측정
>  Importatnt conclusion:
>  * Topographical mapping in the cortex 
>  * Hierarchical organization
>     - Simple cells, Complex cells, Hypercomplex cells

Fukushima 1980: Neurocognitron. network architecture 
> get idea from Hubel & Wiesel experiment\
> simple sells: modifiable parameters\
> complex cells: perform pooling

LeCun 1998: Gradient-based learning applied to document recognition\
Alec Krizhevsky 2012: gave motern incarnation of Convolution Neural Networks. large amount of data is available<br><br>
### ConvNets are everywhere
image classification, image retrieval, detection, segmentation, face-recognition, classifyt videos, pose recognition, game playing, interpretation and diagnosis 
of medical images, street sign recognition, aerial maps, image captioning \
self-driving cars → powered by GPU<br><br>

---
## Convolutional Neural Networks - first without the brain stuff
### Fully Connected Layer
32x32x3 image → stretch to 3072x1\
<img width="650" alt="스크린샷 2021-05-07 오후 8 13 55" src="https://user-images.githubusercontent.com/67621291/117441690-c1f21d80-af70-11eb-9a0d-de36aca14158.png">
### Convolution Layer
- Definition: slide over the image spatially and compute dot products at every spatial location
- Preserve saptial structure 
- Weights are small filters
- filters always extend the full depth of the input volume (image 32x32x*3* - filter 5x5x*3*)

<img width="600" alt="스크린샷 2021-05-07 오후 8 26 48" src="https://user-images.githubusercontent.com/67621291/117443058-8eb08e00-af72-11eb-8311-00e3c49a648f.png"><br><br>

**Activation map**: value of that filter at every spatial location. work with multiple filters.\
<img width="500" alt="스크린샷 2021-05-07 오후 9 29 31" src="https://user-images.githubusercontent.com/67621291/117449781-519cc980-af7b-11eb-8cae-6c72c88abbdc.png">
  <img width="450" alt="스크린샷 2021-05-07 오후 9 30 04" src="https://user-images.githubusercontent.com/67621291/117449831-65483000-af7b-11eb-8bbb-eb520ef1502a.png"><br><br>
Multiple layers stacked together in ConvNet, you end up learning this hierarching of filters.<br> Simple to more complex features.\
One grid is one neuron. Each grid is showing what in the input would look like that basically maximizes the activation of the neuron. → What is the neuron looking for?\
<img width="600" alt="스크린샷 2021-05-07 오후 9 37 13" src="https://user-images.githubusercontent.com/67621291/117450590-64fc6480-af7c-11eb-9d45-858276cb9043.png"><br><br>
**Total Convolution Neural Network**\
<img width="700" alt="스크린샷 2021-05-07 오후 9 50 00" src="https://user-images.githubusercontent.com/67621291/117451975-2e274e00-af7e-11eb-8d0d-a8ed8f2e6213.png">\
Convolution layer → Non-linear layer → ... → Pooling layer → ... → Fully connected layer → Score fuction<br><br>

<img width="250" alt="스크린샷 2021-05-07 오후 9 56 54" src="https://user-images.githubusercontent.com/67621291/117452785-25834780-af7f-11eb-8d60-257f176b4b6c.png"><img width="200" alt="스크린샷 2021-05-07 오후 9 58 46" src="https://user-images.githubusercontent.com/67621291/117453021-6a0ee300-af7f-11eb-99c9-bc40a1176ccf.png">\
If stride = 3 → DOESN'T FIT<br><br><br>
#### Zero pad the border
: to make the size work out to what we want it to\
Deep Network에서 activation map size을 빨리 줄이면 information(= original image를 represent)을 잃는 것\
<img width="200" alt="스크린샷 2021-05-07 오후 10 22 50" src="https://user-images.githubusercontent.com/67621291/117456075-c4f60980-af82-11eb-9830-aabeda9bcf88.png">
<img width="400" alt="스크린샷 2021-05-07 오후 10 23 22" src="https://user-images.githubusercontent.com/67621291/117456141-da6b3380-af82-11eb-8989-c7b709e61e76.png">
➡️ output: ((7 + 1 x 2) - 3) / 1 + 1 = 7x7<br><br>
**EXAMPLES**\
input volume: 32x32x3\
10 5x5 filters with stride 1, pad 2\
`Q) output volume size`: (32+2x2 - 5) / 1 + 1 = 32 spatially (32x32 for each filter)\
➡️ 32x32x10\
`Q) Number of each parameters`: each filter has 5x5x3 + 1 = 76 params (+1은 bias)\
➡️ 76x10 = 760<br><br>
**Common settings**\
Number of filters K : power of 2 (32, 64, 128, 512...)\
their spatial extent F, stride S, the amout of zero padding P\
F = 3, S = 1, P = 1\
F = 5, S = 1, P = 2\
F = 5, S = 2, P = (whatever fits)\
F = 1, S = 1, P = 0<br><br>
### 1x1 convolution layer
<img width="550" alt="스크린샷 2021-05-07 오후 11 31 52" src="https://user-images.githubusercontent.com/67621291/117465196-6897e780-af8c-11eb-8037-9dfe9d13d09a.png"><br>

### CONV layer in Torch
<img width="500" alt="스크린샷 2021-05-07 오후 11 35 06" src="https://user-images.githubusercontent.com/67621291/117465641-dcd28b00-af8c-11eb-8240-bcdce30b9c20.png"><br><br>

### The brain/neuron view of CONV Layer
Differences : neuron has local connectivity \
<img width="200" alt="스크린샷 2021-05-08 오전 1 00 19" src="https://user-images.githubusercontent.com/67621291/117477273-c3cfd700-af98-11eb-800e-4d0853eb8598.png">
<img width="300" alt="스크린샷 2021-05-08 오전 1 00 48" src="https://user-images.githubusercontent.com/67621291/117477320-d518e380-af98-11eb-9171-01f00d9e9c98.png">\
An activation map is a 28x28 sheet of neuron\
output:
1. Each is connected to a small region in the input
2. All of them share parameters

5x5 filter → 5x5 receptive field for each neuron <br><br>
<img width="300" alt="스크린샷 2021-05-08 오전 1 05 15" src="https://user-images.githubusercontent.com/67621291/117477861-74d67180-af99-11eb-8a46-a3890fb68a65.png">
▷ There will be 5 different neurons all looking at the same region in the input volume\
Fully connected layer → each neuron looks at the full input volume <br><br><br>
### Pooling Layer
- makes the representations smaller and more manageable
- operates over each activation map independently
- only pooling spatially. doesn't do anything in the depth. (input depth = output depth)

<img width="300" alt="스크린샷 2021-05-08 오전 1 14 04" src="https://user-images.githubusercontent.com/67621291/117478896-af8cd980-af9a-11eb-8dd7-3e7078d9c81a.png"><br><br>
#### Max pooling
overlap 안되게 하는게 common\
typically don't use zero padding at pooling layer. directly downsample\
<img width="500" alt="스크린샷 2021-05-08 오전 1 15 42" src="https://user-images.githubusercontent.com/67621291/117479137-e9f67680-af9a-11eb-919c-849528ebaebb.png"><br><br>
**Common settings**\
their spatial extent F, stride S\
F = 2, S = 2\
F = 3, S = 2
