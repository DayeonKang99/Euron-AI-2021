# Backpropagation and Neural Networks
### Computational Graph
<img width="500" alt="스크린샷 2021-05-02 오후 7 04 11" src="https://user-images.githubusercontent.com/67621291/116809483-309e3800-ab79-11eb-8f6e-3e33a3d8add9.png">\
Useful for 'Convolutional network (AlexNet)', 'Neural Turing Machine'
<br><br>
## Backpropagation
Break down given f into computational nodes using computational graph. Then you can work with gradients of very simple computations. \
→ Use 'Chain Rule' and get value of gradient<br>
Direction : end of graph → beginning. <br><br>
`1st step)` Calculate in forward\
`2nd step)` The gradient with respect to output is just 1. (맨 마지막 단의 gradient는 1)\
`3rd step)` At each node, find the gradients with respect to just before the node 
(Chain rule = gradients coming from upstream * local gradient). \
`4th step)` Send back to the connected nodes, the next nodes going backwards.<br><br>
<img width="600" alt="스크린샷 2021-05-02 오후 7 41 08" src="https://user-images.githubusercontent.com/67621291/116810364-58dc6580-ab7e-11eb-9550-05726f2f4928.png"><br><br>
#### Sigmoid Fuction
<img width="150" alt="스크린샷 2021-05-02 오후 8 12 42" src="https://user-images.githubusercontent.com/67621291/116811167-c25e7300-ab82-11eb-9c50-4c415fcb03d4.png">\
<img width="500" alt="스크린샷 2021-05-02 오후 8 18 38" src="https://user-images.githubusercontent.com/67621291/116811296-98f21700-ab83-11eb-99db-ee39d5295621.png"><br><br>
#### *Example of Backpropagation and Apply Sigmoid*
<img width="250" alt="스크린샷 2021-05-02 오후 8 21 24" src="https://user-images.githubusercontent.com/67621291/116811379-f8e8bd80-ab83-11eb-8b39-e04ee3cf6670.png">
<img width="650" alt="스크린샷 2021-05-02 오후 8 21 48" src="https://user-images.githubusercontent.com/67621291/116811391-0736d980-ab84-11eb-86ac-f710c2072fcb.png">
<br><br>
### Pattern in backward flow
**Add** gate : gradient distributor. pass same thing to both of the branches that were connected.\
**Max** gate : gradient router. one gets the full value of the gradient just passed back, and the other one will have a gradient of zeor.\
**Mul** gate : gradien switcher and scaler. take upstream gradient and scale it by the value of the other branch.<br><br>
**Gradients add at branches**\
<img width="250" alt="스크린샷 2021-05-02 오후 8 35 49" src="https://user-images.githubusercontent.com/67621291/116811768-fc7d4400-ab85-11eb-9afe-065e47fe9927.png">
 ▷ Sum these up to be the total upstream gradient flowing back into this node.<br><br>
 
### Gradients for vectorized code
Gradient is going to be **Jacobian matrix**\
`Jacobian matrix` : derivative of each element of z with respect to each element of x (example)<br><br>
<img width="400" alt="스크린샷 2021-05-02 오후 8 51 49" src="https://user-images.githubusercontent.com/67621291/116812157-394a3a80-ab88-11eb-80d4-ecd830ec1784.png">
  <img width="200" alt="스크린샷 2021-05-02 오후 8 53 29" src="https://user-images.githubusercontent.com/67621291/116812205-744c6e00-ab88-11eb-8e8c-8d15eae7d160.png">\
In this example, \
`1)` Jacobian matrix size = [4096 x 4096]\
> In practice, we process an entire minibatch (e.g. 100) of examples at one time \
> : Jacobian would technically be a [409600 x 409600] matrix → ❌ (really huge, completely impractical)

`2)` Jacobian matrix look like diagonal 
> Each element of the input only affects that corresponding element in the output. → diagonal

<br>
<img width="750" alt="스크린샷 2021-05-02 오후 9 28 28" src="https://user-images.githubusercontent.com/67621291/116813153-57ff0000-ab8d-11eb-8fb4-64849fd4761c.png">
Each element of your gradient is quantifiying how much that element is affecting your final output.<br><br>

### Modularized implementation: forward / backward API
``` python
class ComputationalGraph(object):
  def forward(inputs):
    #1. [pass inputs to input gates...]
    #2. forward the computational graph:
    for gate in self.graph.nodes_topologically_sorted():
      gate.forward()
    return loss # the fianl gate in the graph outputs the loss
  def backward():
    for gate in reversed(self.graph.nopdes_topologically_sorted()):
      gate.backward() # little piece of backprop (chain rule applied)
    return inputs_gradients
```
``` python
class MultiplyGate(object):
  def forward(x, y):
    z = x * y
    self.x = x
    self.y = y
    return z
  def backward(dz):
    dx = self.y * dz # [dz/dx * dL/dz]
    dy = self.x * dz # [dz/dy * dL/dz]
    return [dx, dy]
```
<br><br>

---

## Neural Networks
**Neural Network: without the brain stuff**\
<img width="500" alt="스크린샷 2021-05-02 오후 10 52 21" src="https://user-images.githubusercontent.com/67621291/116815565-0f4d4400-ab99-11eb-89c0-bb4884f42936.png">\
w1 can be many different kind of templates. w2 weighted sum of all of these templates. h is like score function of w1. \
non-linear thing is right before h. linear function is on top.<br><br>
Stack more layers to get deeper networks → 3-layer Neural Network *f = W3 max(0, W2 max(0, w1x))*<br><br>
**Full implementation of training 2-layer Neural Network - Simple**
``` python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
  h = 1 / (1 + np.exp(-x.dot(w1)))
  y_pred = h.dot(w2)
  loss = np.square(y_pred - y).sum()
  print(t, loss)
  
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h.T.dot(grad_y_pred)
  grad_h = grad_y_pred.dot(w2.T)
  grad_w1 = x.T.dot(grad_h * h * (1 - h((
  
  w1 -= 1e-4 * grad_w1
  w2 -= 1e-4 * grad_w2
```
<br><br>
<img width="700" alt="스크린샷 2021-05-02 오후 11 12 39" src="https://user-images.githubusercontent.com/67621291/116816184-e5e1e780-ab9b-11eb-8bcd-fd3a1e9096a2.png">\
non-linearities are activation function\
(The most similar thing to what neurons behave is the ReLU function.)<br><br>
### Biological Neurons
- Many different types
- Dendrites can perform complex non-linear computations
- Synapses are not a single weight but a complex non-linear dynamical system
- Rate code may not be adequate

**Be very careful with your brain analogies**\
In practice, neuron is much more complicated.<br><br><br>
<img width="500" alt="스크린샷 2021-05-02 오후 11 27 39" src="https://user-images.githubusercontent.com/67621291/116816705-feeb9800-ab9d-11eb-8bfe-1fbbb1f1eaed.png">
<br><br>
<img width="550" alt="스크린샷 2021-05-02 오후 11 28 19" src="https://user-images.githubusercontent.com/67621291/116816722-1591ef00-ab9e-11eb-8406-c2ae5a857e9a.png">\
**Forward-pass of a 3-layer neural network:**
```python
f = lambda x: 1.0 / (1.0 + np.exp(-x) # activation function (sigmoid)
x = np.random.randn(3, 1) # input vector (3x1)
h1 = f(np.dot(W1, x) + b1) # first hidden layer (4x1)
h2 = f(np.dot(W2, h1) + b2) # second hidden layer (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)
```
