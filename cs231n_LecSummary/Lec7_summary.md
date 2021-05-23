# Training Neural Networks II
## Optimization
### Problems with SGD
What if loss changes quickly in one direction and slowly in another?<br>What does gradient descent do?<br>
➡️ Very slow progress along shallow dimension, jitter along steep direction (much more common in high dimension)<br><br>
What if the loss fuction has a local minima or saddle point?<br>
➡️ Zero gradient, gradient descent gets stuck. Saddle points much more common in high dimension (saddle point 근처, slope가 매우 작은 곳도 문제)<br><br>
Whole training set is very large, so computing loss is very expensive. In practice, estimate loss and gradient using a small mini batch<br>
➡️ We're not actually getting the true information about the gradient at every time step. We're just getting some noisy. (take a long time)
<br><br>
### SGD + Momentum
<img width="600" alt="스크린샷 2021-05-23 오전 2 15 45" src="https://user-images.githubusercontent.com/67621291/119235305-c91e4b80-bb6c-11eb-87a1-5c5ae0eac74f.png">
- Build up "velocity" as a running mean of gradients<br>
- Rho gives "friction"; typically rho=0.9 or 0.99<br>
(velocity가 있어서 공이 언덕을 내려올수록 속도가 빨라지는 것과 같음)<br><br>
<img width="550" alt="스크린샷 2021-05-23 오전 2 30 25" src="https://user-images.githubusercontent.com/67621291/119235712-d805fd80-bb6e-11eb-8a76-41a64b8e5b05.png">

`Normal SGD Momentum`:<br>
estimate the gradient at current point, and then take a mix of velocity and gradient<br>
`With Nesterov Momentum`:<br>
step in the direction of where the velocity would take you. evaluate the gradient at that point. go back to your original point, and mix together those two. 
<br><br>

### Nesterov Momentum
<img width="700" alt="스크린샷 2021-05-23 오전 3 11 44" src="https://user-images.githubusercontent.com/67621291/119236860-9bd59b80-bb74-11eb-89bb-aa311acb32ee.png">
<br>

### AdaGrad
``` python
grad_squared = 0
while True:
  dx = compute_gradient(x)
  grad_squared += dx * dx  # Added element-wise scaling of the gradient based on the historical sum of squares in each dimension
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```
`Q)` What happens with AdaGrad?<br>
`A)` 2 cordinate이고 하나는 항상 high gradient, 다른 하나는 항상 small gradient이면, small gradient의 제곱을 더하고 divide하면 accelerate movement along the one dimension. 
다른 dimension은 gradient가 크고 큰 수로 나누면 slow down our progress along the wiggling dimension.<br>
`Q)` What happens to the step size over long time?<br>
`A)` with AdaGrad, the steps actually get smaller and smaller. squared gradient를 계속 업데이트하기 때문에 계속 커져서 step size를 계속 작게 만듦. <br>
→ continually decay learning rates, get stuck 
<br><br>
### RMSProp
<img width="650" alt="스크린샷 2021-05-23 오후 12 20 07" src="https://user-images.githubusercontent.com/67621291/119247005-38be2600-bbc1-11eb-883d-0437d576c6df.png">
RMSProp adjust its trajectory (SGD + Momentum은 overshooting 하는데 반해)
<br><br>

### Adam
<img width="900" alt="스크린샷 2021-05-23 오후 12 46 37" src="https://user-images.githubusercontent.com/67621291/119247497-ec74e500-bbc4-11eb-93a9-fb05a051d854.png">
Bias correction for the fact that first and second moment estimates start at zero (→ first step이 매우 커지는 경우 발생)<br>

**Adam with beta1 = 0.9, beta2 = 0.999, <br>and learning_rate = 1e-3 or 5e-4<br>is a great starting point for many models!**<br><br>
<img width="500" alt="스크린샷 2021-05-23 오후 12 52 16" src="https://user-images.githubusercontent.com/67621291/119247585-b6843080-bbc5-11eb-9b4f-d8999653af10.png">
<br><br><br>

### Learning rate Decay
<img width="300" alt="스크린샷 2021-05-23 오후 2 59 41" src="https://user-images.githubusercontent.com/67621291/119249915-85146080-bbd7-11eb-815d-8cc067f3dfa3.png"> ➡️ Learning rate decay over time
<br>
step decay:<br>e.g. decay learning rate by half every few epochs<br>
<img width="200" alt="스크린샷 2021-05-23 오후 3 01 50" src="https://user-images.githubusercontent.com/67621291/119249962-d02e7380-bbd7-11eb-8cdc-be3bf8bb9508.png">\
<img width="300" alt="스크린샷 2021-05-23 오후 3 02 18" src="https://user-images.githubusercontent.com/67621291/119249972-e0dee980-bbd7-11eb-877f-4b015b0fc1a8.png">\
Learning rate decay is more common with SGD momentum, less common with Adam<br>
Learning rate decay is second-order hyperparameter (처음에는 learning rate decay 없이 시작)<br><br>

### First-Order Optimization
<img width="580" alt="스크린샷 2021-05-23 오후 3 09 44" src="https://user-images.githubusercontent.com/67621291/119250131-ea1c8600-bbd8-11eb-8c40-460e3900b662.png"><br><br>

### Second-Order Optimization
<img width="600" alt="스크린샷 2021-05-23 오후 3 10 15" src="https://user-images.githubusercontent.com/67621291/119250144-fd2f5600-bbd8-11eb-9750-205cff50ac4d.png">
<br>
<img width="600" alt="스크린샷 2021-05-23 오후 3 13 35" src="https://user-images.githubusercontent.com/67621291/119250212-75961700-bbd9-11eb-98e7-2ba6a891c71c.png">

➡️ Learning rate 필요 없음\
➡️ Impractical. 
> Hessian has O(N^2) elements\
> Inverting takes O(N^3)\
> N = (Tens or Hundreds of) Millions

#### Quasi-Newton methods (BGFS most popular)
instead of inverting the Hessian (O(n^3)), approximate inverse Hessian with rank 1 updates over time (O(n^2) each)
<br>
#### L-BFGS (Limited memory BFGS)
Des not form/store the full inverse Hessian<br>It does not work too well for many deep learning problems
<br><br><br>

## Beyond Training Error
We really care about reducing gap between train and test error<br>

### Model Ensembles 
1. Train multiple independent models
2. At test time average their results

2% extra performance <br><br>
**Tips and Tricks**<br>
- Instead of training independent models, use multiple snapshots of a single model during training 
- Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time (Polyak averaging)

<br><br>
How to improve single-model performance?<br>
➡️ Regularization
<br><br>

### Regularization: Add term to loss 
<img width="600" alt="스크린샷 2021-05-23 오후 11 51 50" src="https://user-images.githubusercontent.com/67621291/119265469-da755f80-bc21-11eb-86b7-fe035083ab6d.png">
<br>

### Regularization: Dropout
In each forward pass, randomly set some neurons to zero <br>Probability of dropping is a hyperparameter; 0.5 is common<br>
<img width="450" alt="스크린샷 2021-05-23 오후 11 55 44" src="https://user-images.githubusercontent.com/67621291/119265650-65eef080-bc22-11eb-910e-3222766f6094.png">
<img width="500" alt="스크린샷 2021-05-24 오전 12 04 44" src="https://user-images.githubusercontent.com/67621291/119265933-a733d000-bc23-11eb-9a00-682f9aca7dec.png"><br><br>
Forces the network to have a redundant representation;<br>Prevents co-adaptationn of features<br><br>
Another interpretation: <br>Dropout is training a large **ensenble** of models (that share parameters) <br>Each binary mask is one model<br><br>
#### Dropout: Test time
<img width="300" alt="스크린샷 2021-05-24 오전 12 53 59" src="https://user-images.githubusercontent.com/67621291/119267652-87ec7100-bc2a-11eb-888c-bc7d3a252f58.png"><br><br>
<img width="650" alt="스크린샷 2021-05-24 오전 12 57 29" src="https://user-images.githubusercontent.com/67621291/119267791-06491300-bc2b-11eb-985e-29cf1b022bf7.png">
<br><br>
**Summary**<br>
<img width="700" alt="스크린샷 2021-05-24 오전 1 05 13" src="https://user-images.githubusercontent.com/67621291/119268051-19a8ae00-bc2c-11eb-90c6-098416acb0c1.png">
<br><br>

#### More common: Inverted Dropout
<img width="700" alt="스크린샷 2021-05-24 오전 1 06 16" src="https://user-images.githubusercontent.com/67621291/119268081-4066e480-bc2c-11eb-91fa-22a4fafd0ac4.png">
<br>

### Regularization: Data Augmentation
Randomly transfrom the image, and train
- Horizontal flips
- Random crops and scales
- Color jitter 
- ...

<br>

### Regularization: A common pattern
**Training**: Add some kind of randomness<br>
**Testing**: Average out randomness (sometimes approximate)<br><br>
Batch Normalization - regularization pattern 가짐<br>
**Training**: Normalize using stats from random minibatches<br>
**Testing**: Use fixed stats to normalize<br>
→ batch normalization 사용하면 dropout 사용하지 않음
<br>

#### Examples:
- Dropout
- Batch Normalization
- Data Augmentation
- DropConnect 
- Fractional Max Pooling
- Stochastic Depth

<br><br>

## Transfer Learning
CNN을 사용하려면 data가 많아야한다 → ❌ **BUSTED**<br><br>
<img width="800" alt="스크린샷 2021-05-24 오전 1 32 22" src="https://user-images.githubusercontent.com/67621291/119268882-e536f100-bc2f-11eb-9d33-3668bfa1349a.png"><br><br>
<img width="400" alt="스크린샷 2021-05-24 오전 1 34 34" src="https://user-images.githubusercontent.com/67621291/119268960-33e48b00-bc30-11eb-9359-c27cab814202.png"><br><br>
**Transfer learning with CNNs is pervasive (it's the norm, not an exception)**<br>
<img width="700" alt="스크린샷 2021-05-24 오전 1 40 32" src="https://user-images.githubusercontent.com/67621291/119269120-09470200-bc31-11eb-8b21-910304425b2d.png">
<br><br><br>
large data set을 가지고 있지 않으면 <br>→ download some pretrained model that's relatively close to the task you care about, and then either reinitialize parts of that model or fine tune that model for your data<br>
Deep learning frameworks provide a "Model Zoo" of pretrained models so you don't need to train your own
