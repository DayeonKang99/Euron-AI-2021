# Recurrent Neural Networks
<img width="919" alt="스크린샷 2021-07-01 오전 10 22 32" src="https://user-images.githubusercontent.com/67621291/124050817-408da780-da56-11eb-9274-71b234b75fde.png"><br>
one to many : Image Captioning (image → sequence of words)<br>
many to one : Sentiment Classificaiton (sequence of words → sentiment)<br>
many to many : Machine Translation (seq of words → seq of words)<br>
many to many : video classification on frame level <br><br><br>

<img width="750" alt="스크린샷 2021-07-01 오전 10 43 10" src="https://user-images.githubusercontent.com/67621291/124052304-22757680-da59-11eb-9f2c-0dddc03d4131.png"><br>
x가 input으로 들어오고 RNN은 internal hidden state를 가지고 있어 update every time. <br>Internal hidden state는 모델로 fed back 되고 다음에 input으로 read. And produce an output<br>
additional fully connected layer가 every time step마다 ht를 읽어서 output을 produce<br>
**Notice:** the same function and the same set of parameters are used at every time step <br><br>

### Vanilla Recurrent Neural Network
<img width="400" alt="스크린샷 2021-07-01 오전 10 47 17" src="https://user-images.githubusercontent.com/67621291/124052593-b5aeac00-da59-11eb-9e05-545522e6358c.png"><br>
<br>
### Computational Graph
unique x, h, but re-use the same weight matrix<br>
**Many to Many**<br>
<img width="600" alt="스크린샷 2021-07-01 오전 10 56 26" src="https://user-images.githubusercontent.com/67621291/124053314-fce96c80-da5a-11eb-8295-0baff4321ee1.png"><br>

**Many to One**<br>
<img width="550" alt="스크린샷 2021-07-01 오전 10 57 03" src="https://user-images.githubusercontent.com/67621291/124053377-12f72d00-da5b-11eb-9f06-9fb435cf3e82.png"><br>

**One to Many**<br>
<img width="550" alt="스크린샷 2021-07-01 오전 10 57 42" src="https://user-images.githubusercontent.com/67621291/124053420-2a361a80-da5b-11eb-8f07-5e370c201d2b.png"><br>

**Sequence to Sequence: many-to-one(encoder) + ont-to-many(decoder)**<br>
many-to-one: summarize the variably sized of input<br>
<img width="700" alt="스크린샷 2021-07-01 오전 11 02 40" src="https://user-images.githubusercontent.com/67621291/124053815-dbd54b80-da5b-11eb-9fa9-cfd314e223e7.png"><br>
<br>

**example) Character-level Language Model** <br>
<img width="450" alt="스크린샷 2021-07-01 오전 11 06 41" src="https://user-images.githubusercontent.com/67621291/124054147-6c139080-da5c-11eb-9eee-6368dd63ba4f.png"><br>
first step에서 다음에 올 character로 o 예측 → compute loss<br>
at every time step, predict the next character <br><br>
**Character-level Language Model Sampling** <br>
at test-time sample characters one at a time, feed back to model <br>
<img width="421" alt="스크린샷 2021-07-01 오전 11 11 41" src="https://user-images.githubusercontent.com/67621291/124054526-1ee3ee80-da5d-11eb-8155-e325be0296b9.png"><br>
<br><br>
### Truncated Backpropagation through time 
Run forward and backward through chunks of the sequence instead of whole sequence <br>
Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps <br>
<img width="500" alt="스크린샷 2021-07-01 오후 7 21 04" src="https://user-images.githubusercontent.com/67621291/124108826-7c9a2a00-daa1-11eb-9e45-dcaf06f8c2be.png">
<br><br><br>
## Image Captioning 
convolution neural network(CNN) produce a summary vector of the image which will then feed into the first time step of recurrent neural network(RNN) model. <br>
<img width="600" alt="스크린샷 2021-07-01 오후 7 44 24" src="https://user-images.githubusercontent.com/67621291/124111904-bf113600-daa4-11eb-98f5-81c91b561571.png"><br>
<br>
`At test time,` there are no FC-1000 and softmax in CNN. <br>At the end of the model, FC-4096 used to summarze the whole content of the image.<br>
At RNN model, we need first initial input to tell it to start generating text. In this case, we'll give it some special start token.<br>
<img width="600" alt="스크린샷 2021-07-01 오후 10 54 19" src="https://user-images.githubusercontent.com/67621291/124135969-479cd000-dabf-11eb-9f35-7b8ef42d9d4e.png"><br>
<br>
## Image Captioning with Attention
<img width="742" alt="스크린샷 2021-07-01 오후 11 00 36" src="https://user-images.githubusercontent.com/67621291/124136898-27214580-dac0-11eb-979b-804c41ac9c41.png"><br>
<br><img width="800" alt="스크린샷 2021-07-01 오후 11 03 33" src="https://user-images.githubusercontent.com/67621291/124137343-90a15400-dac0-11eb-8407-99ce3316f696.png"><br>
attention is shifting around different parts of the image for each word in the caption that it generates. <br>
it tends to focus it's attention on the salient or semanticly meaningful part of the image when generating captions.<br><br>

## Visual Question Answering: RNNs with Attention
asking questions about the image<br>
RNN으로 input question을 single vector로 summarize + CNN으로 summarize the image → combine and predict a distribution over answers<br>
<img width="831" alt="스크린샷 2021-07-01 오후 11 19 11" src="https://user-images.githubusercontent.com/67621291/124139620-c0e9f200-dac2-11eb-94c2-c6066b4254e5.png">
<br><br><br>

## Multilayer RNNs
2, 3, 4 layers RNNs. 깊게 layer를 쌓지는 않는다<br>
<img width="300" alt="스크린샷 2021-07-02 오전 12 14 43" src="https://user-images.githubusercontent.com/67621291/124148320-81bf9f00-daca-11eb-93f9-bc5875844b87.png"><br>
<br>

## Vanila RNN Gradient Flow
backpropagation from ht to ht-1 multiplies by W (actually Whh.T)<br>
computing gradient of h0 involves many factors of W (and repeated tanh)<br>
<img width="909" alt="스크린샷 2021-07-02 오전 12 11 23" src="https://user-images.githubusercontent.com/67621291/124147859-0b22a180-daca-11eb-9e90-1ab35d90644d.png"><br>
같은 수를 계속해서 곱하면<br>
Largest singular value < 1: **Vanishing gradients** → change RNN architecture <br>
Largest singular value > 1: **Exploding gradients** (h0의 gradient) → **Gradient clipping**<br>
> Gradient clipping: Scale gradient if its norm is too big <br>
> ``` python
> grad_norm = np.sum(grad * grad)
> if grad_norm > threshold:
>   grad *= (threshold / grad_norm)
> ```
<br>

## Long Shor Term Memory (LSTM)
maintain 2 hidden states at every time step (ht, ct(:cell state, internal))<br>
<img width="300" alt="스크린샷 2021-07-02 오전 12 17 15" src="https://user-images.githubusercontent.com/67621291/124148700-dcf19180-daca-11eb-9a44-1a6ef9a6149e.png"><br>
<br>
hidden state, current input, and then use those to compute four gates(i, f, o, g)<br>
sigmoid: value will be btw 0 ~ 1<br>
tanh: value will be btw -1 ~ 1<br>
cell state as being little scaler integer counters (증가 / 감소 at time step)<br>
after compute cell state, update cell state to compute hidden state which we will reveal to the outside world (at output gate)<br><br>
<img width="800" alt="스크린샷 2021-07-02 오전 12 22 24" src="https://user-images.githubusercontent.com/67621291/124149476-96506700-dacb-11eb-87b6-aa1b22a5c772.png"><br>
<img width="400" alt="스크린샷 2021-07-02 오전 12 32 45" src="https://user-images.githubusercontent.com/67621291/124150966-07444e80-dacd-11eb-9f88-5ccc102e39f9.png"><br><br>
backpropagation from ct to ct-1 only elementwise multiplication by f, no matrix multiply by W (f는 time step마다 다름, sigmoid라 0~1을 guarantee)<br>
<img width="800" alt="스크린샷 2021-07-02 오전 1 17 07" src="https://user-images.githubusercontent.com/67621291/124156981-3958af00-dad3-11eb-8de3-ccfb09ae0e77.png"><br><br>

## Other RNN Variants
**GRU**: Learning phrase representations using RNN encoder-decoder for statistical machine translation, Cho et al.2014<br>
LSTM: A Search Space Odyssey, Greff et al.,2015<br>
An Empirical Exploration of Recurrent Network Architectures, Jozefowicz et al.,2015
