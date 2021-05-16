# Training Neural Networks I
## Activation Functions
### Sigmoid 
<img width="200" alt="스크린샷 2021-05-16 오후 6 53 55" src="https://user-images.githubusercontent.com/67621291/118393087-133c9400-b678-11eb-86b9-caa25880cb5c.png"><img width="200" alt="스크린샷 2021-05-16 오후 6 24 09" src="https://user-images.githubusercontent.com/67621291/118392301-e9816e00-b673-11eb-9654-c4479c1c2d33.png">
- Squashes numbers to range [0,1]
- Historically popular since they have nice interpretation as a saturating 'firing rate' of a neuron

**Problems**
1. Saturated(포화된) neurons 'kill' the gradients (X가 매우 큰 음수이거나 양수이면 sigmoid가 flat, gradient를 0으로 만들기 때문)
2. Sigmoid outputs are not zero-centered <br>(input이 모두 양수이거나 음수이면 가중치의 gradient가 항상 move in the same direction. W가 increase만 / decrease만 → inefficient)
3. exp() is a bit compute expensive → not the main problem
<br>

### tanh(x)
<img width="200" alt="스크린샷 2021-05-16 오후 6 52 44" src="https://user-images.githubusercontent.com/67621291/118393054-e7b9a980-b677-11eb-8186-cc71d77260f1.png">

- Squashes numbers to range [-1, 1]
- zero centered 🙂
- still kill gradients when saturated ☹️ (flat하기 때문)
<br>

### ReLU (Rectified Linear Unit)
<img width="200" alt="스크린샷 2021-05-16 오후 6 59 34" src="https://user-images.githubusercontent.com/67621291/118393210-dc1ab280-b678-11eb-9444-8b807c38c22c.png"><img width="150" alt="스크린샷 2021-05-16 오후 7 00 00" src="https://user-images.githubusercontent.com/67621291/118393219-eb99fb80-b678-11eb-883b-ba5cf130457a.png">

- Does not saturate (in + region)
- Very computationally efficient
- Converges(수렴) much faster than sigmoid/tanh in practice (6배 빠름)
- Actually more biologically plausible than sigmoid
- 2012년 AlexNet에서 사용

**Problems**
1. Not zero0centered output
2. An annoyance: x < 0 일 때 saturate 
<img width="400" alt="스크린샷 2021-05-16 오후 7 08 19" src="https://user-images.githubusercontent.com/67621291/118393432-15075700-b67a-11eb-94b3-a3059e356fb0.png">
➡️ initialize 잘못하면 never update / learning rate is too high이면 나빠질 수 있고 die 가능 <br>
➡️ people like to initialize ReLU neurons with slightly positive biases to get updates (그러나 많은 사람들은 0 bias로 initialize)<br><br>

### Leaky ReLU
<img width="200" alt="스크린샷 2021-05-16 오후 7 51 15" src="https://user-images.githubusercontent.com/67621291/118394507-1471bf00-b680-11eb-9337-a99ac04e9388.png"> <img width="200" alt="스크린샷 2021-05-16 오후 7 51 44" src="https://user-images.githubusercontent.com/67621291/118394525-26536200-b680-11eb-97cc-460af0e102ee.png">
- Does not saturate
- Computationally efficient
- Converges much faster than sigmoid/tanh in practice
- **Will not 'die'**

### PReLU (Parametric Rectifier)
<img width="230" alt="스크린샷 2021-05-16 오후 7 56 26" src="https://user-images.githubusercontent.com/67621291/118394640-cdd09480-b680-11eb-80ee-217712078e99.png">
Just like a leaky ReLU. We don't hard-code it. Treat it as a parameter.<br>
backprop into alpha (parameter) → it gives a little bit more flexibility <br><br>

### ELU (Exponential Linear Units)
<img width="200" alt="스크린샷 2021-05-16 오후 8 12 51" src="https://user-images.githubusercontent.com/67621291/118395017-18531080-b683-11eb-8971-099fac25022b.png"> <img width="300" alt="스크린샷 2021-05-16 오후 8 13 15" src="https://user-images.githubusercontent.com/67621291/118395028-2739c300-b683-11eb-85a3-56fcdaf589b7.png">
- All benefits of RuLU
- Closer to zero mean outputs
- Negative saturation regime compared with Leaky ReLU adds some robustness to noise

**Problem**<br>Computation requires exp()<br><br>

### Maxout "Neuron"
<img width="300" alt="스크린샷 2021-05-16 오후 8 17 25" src="https://user-images.githubusercontent.com/67621291/118395145-bc3cbc00-b683-11eb-92bc-4f378835c36d.png">

- Does not have the basic form of dot product → nonlinearity
- Generalizes ReLU and Leaky ReLU
- Linear Regime! Does not saturate! Does not die!

**Problem**<br>doubles the number of parameters/neuron (W1, W2)<br><br>

### In practice
- Use ReLU. Be careful with your learning rates
- Try out Leaky ReLU / Maxout / ELU
- Try out tanh but don't expect much
- Don't use sigmoid

<br><br>
## Data Preprocessing
### Step 1: Preprocess the data
<img width="620" alt="스크린샷 2021-05-16 오후 8 38 06" src="https://user-images.githubusercontent.com/67621291/118395684-a11f7b80-b686-11eb-9c23-4f24bb47f4f7.png"><img width="380" alt="스크린샷 2021-05-16 오후 8 45 26" src="https://user-images.githubusercontent.com/67621291/118395872-a5986400-b687-11eb-9a9a-9d2b1f75a0de.png">
<br><br>
input이 모두 양수이면 weights의 gradient도 positive. 모두 0이거나 음수가 아니더라도 bias가 이런 문제 야기.<br>
이미지는 보통 zero centering은 하지만 normalize, PCA, Whitening은 많이 하지 않음 (PCA, Whitening → more complicate)<br><br>
### In practice for imgaes: center only
e.g. consider CIFAR-10 example with [32, 32, 3] images
- Subtract the mean image (e.g. AlexNet) <br>mean image = [32, 32, 3] array
- Subtract per-channel mean (e.g. VGGNet) <br>mean along each channel = 3 numbers

**do the same thing at test time** for this array that you determined at training time
<br><br><br>

## Weight Initialization
`Q)` What happens when W = 0 init is used?<br>
`A)` Every neuron have the same operation. same output → same gradient → update in the same way → all neurons are exactly the same<br><br>
`First idea` Small random numbers (gaussian with zero mean and 1e-2 standard deviation)<br>
```python
W = 0.01 * np.random.randn(D, H)
```
**Probelm:** works okay for small networks, but problems with deeper networks<br>
각 layer에 작은 숫자의 W가 반복해서 곱해지면 quickly shrinks and collapse<br><br>
E.g. 10-layer net with 500 neurons on each layer, using tanh non-linearities<br>
`Q)` Think about the backward pass. What do the gradients look like?<br>
`A)` W * X gate를 지나면 X. X가 작으면 W는 small gradient를 얻어 not updating (forward pass와 같은 현상)<br><br>
### 1.0 instead of 0.01
```python
W = 1.0 * np.random.randn(D, H)
```
**Problem:** Almost all neurons completely saturated, either 01 and 1. Gradients will be all zero. Weights are not updating<br><br>
### Xavier initialization
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
```
➡️ smaller number of input → get larger weights (large → smaller)<br>Reasonable initialization. Mathematical derivation assumes linear activations<br>

**Problem:** when using the ReLU nonlinearity it breaks (because it's killing half of units) distributions start collapsing, more units deactivated
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)
```
adjusting for the fact that half the neurons get killed
<br><br><br>

## Batch Normalization
consider a batch of activations at some layer. <br>To make each dimension unit gaussian, apply<br>
<img width="550" alt="스크린샷 2021-05-16 오후 11 11 17" src="https://user-images.githubusercontent.com/67621291/118400304-06319c00-b69c-11eb-8674-0bd35ea4cace.png">\
Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.<br>activation map 당 normalize\
<img width="130" alt="스크린샷 2021-05-16 오후 11 12 40" src="https://user-images.githubusercontent.com/67621291/118400351-3842fe00-b69c-11eb-933f-3a2fa45f637e.png">\
**Problem:** do we necessarily want a unit gaussian input to a tanh layer?<br>
<img width="300" alt="스크린샷 2021-05-16 오후 11 22 27" src="https://user-images.githubusercontent.com/67621291/118400682-96241580-b69d-11eb-9f79-0e2240b4a2fa.png"><img width="200" alt="스크린샷 2021-05-16 오후 11 22 53" src="https://user-images.githubusercontent.com/67621291/118400692-a50ac800-b69d-11eb-96ab-6cc0096a5c78.png">
<br><br>
<img width="450" alt="스크린샷 2021-05-16 오후 11 25 58" src="https://user-images.githubusercontent.com/67621291/118400784-134f8a80-b69e-11eb-8e6c-9005237944bd.png">
- improves gradient flow through the network
- allows higher learning rates
- reduces the strong dependence on initialization
- acts as a form of regularization in a funny way, and slightly reduces the need for dropout, maybe

**Note: at test time BatchNorm layer fuctions differently**: <br>
The mean/std are not computed based on the batch. Instead, a single fixed empirical mean of activations during training is used.<br>don't re-compute at test time. 
estimate at training time, and then use this at test time
<br><br><br>

## Babysitting the Learning Process
Step 1: Preprocess the data<br>
Step 2: Choose the architecture<br> 
> Double check that the loss is reasonable<br>
> zero regularization의 loss 보다 crank up regularization에서의 loss가 더 커야함

Step 3: Try to train<br> 
> Tip: make sure that you can overfit very small portion of the training data

`Real train` Start with small regularization and find learning rate that makes the loss go down<br>
Loss barely changing: Learning rate is probably too low<br>
> loss가 거의 변하지 않아도 weights가 옳은 방향으로 변하고 있기 때문에 accuracy가 크게 향상될 수 있다

Loss exploding: learning rate too high (NaN)<br>
➡️ Rough range for learning rate we should be cross-validating is somewhere [1e-3 ... 1e-5]
<br><br>

## Hyperparameter Optimization
### Cross-validation strategy
First stage: only a few epochs to get rough idea of what params work<br>
Second stage: longer running time, finer search<br>
... (repeat as necessary)<br><br>
Tip for detecting explosions(NaN) in the solver:<br>
If the cost is ever > 3 * original cost, break out early
<br><br>
### Random Search vs. Grid Search
<img width="500" alt="스크린샷 2021-05-17 오전 12 31 00" src="https://user-images.githubusercontent.com/67621291/118402952-29157d80-b6a7-11eb-83ec-62b888149a07.png">\
green fuction showing where good values are<br>
grid는 3개의 점 밖에 없어 good region이 어디인지 놓침<br>
much more useful signal overall since we have more samples of different values of the important variable
<br><br>
### Hyperparameters to play with:
- network architecture
- learning rate, its decay schedule, update type
- regularization (L2 / Dropout strength)
<br><br>
### Monitor and visualize the loss curve
<img width="700" alt="스크린샷 2021-05-17 오전 12 39 45" src="https://user-images.githubusercontent.com/67621291/118403215-62022200-b6a8-11eb-8b36-2e9d1d640dec.png">
<br>
<img width="400" alt="스크린샷 2021-05-17 오전 12 46 42" src="https://user-images.githubusercontent.com/67621291/118403451-5e22cf80-b6a9-11eb-81bf-37a72fa05313.png">
<br>

### Monitor and visualize the accuracy
<img width="700" alt="스크린샷 2021-05-17 오전 12 43 42" src="https://user-images.githubusercontent.com/67621291/118403334-ef457680-b6a8-11eb-9565-c62cd03f0272.png">
<br>

### Track the ratio of weight updates / weight magnitudes
ratio between the updates and values: ~0.0002 / 0.02 = 0.01 (about okay)<br>
want this to be somewhere around 0.001 or so<br>
너무 크거나 작게 update되면 안됨
