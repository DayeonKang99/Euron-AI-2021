# Loss Functions and Optimization
### Loss function
: A loss function tells how good out current classifier is.\
Loss over the dataset is a sum of loss over examples :\
<img width="300" alt="스크린샷 2021-04-04 오후 6 25 29" src="https://user-images.githubusercontent.com/67621291/113504455-25a0b980-9573-11eb-906c-7bfa06e71e3d.png">
> general formulation and extends even beyond image classification

*x* is image, *y* is integer label\
*L_i* is loss function, *L* is average of losses summed over the entire dataset over each of the N examples<br><br>
`score function:`\
`s = f(x;W)`
<br><br>
### Loss function #1 - Multiclass SVM loss
: Loss *L_i* for any individual example, is a sum over all of the categories, *Y*, except for the true category, *Y_i* (sum of incorrect categories)\
*Sy_i* is the score of the correct category, and *Sj* is the score of incorrect category. \
<img width="353" alt="스크린샷 2021-04-04 오후 6 45 03" src="https://user-images.githubusercontent.com/67621291/113504844-df992500-9575-11eb-809e-0da90e06b2e7.png">
> The graph of x-axis is *Sy_i*, y-axis is loss

As the score for the true category increases, then the loss will go down linearly until we get to above this safety margin. \After the loss will be zero because
we've already correctly classified this example. <br><br>
**Example**\
<img width="350" alt="스크린샷 2021-04-04 오후 7 01 32" src="https://user-images.githubusercontent.com/67621291/113505178-2e47be80-9578-11eb-961b-4b73dcada408.png">
<img width="250" alt="스크린샷 2021-04-04 오후 7 01 59" src="https://user-images.githubusercontent.com/67621291/113505193-3c95da80-9578-11eb-8af6-04d71cb3ac87.png">\
<img width="230" alt="스크린샷 2021-04-04 오후 7 02 33" src="https://user-images.githubusercontent.com/67621291/113505210-51726e00-9578-11eb-926f-6c4ad247f2c1.png">
> Min possible loss is 0\
> Max possible loss is infinity<br><br>
> At initialization W is small so all s = 0. Loss is the number of classes - 1\
> because incorrect classes is the number of classes - 1 and margin (1) will be left.<br><br>
> What if the sum was over all classes? (including *j* = *y_i*) (전에는 틀린 클래스만 더했는데 만약 옳은 클래스도 더한다면?)\
> Loss is increases by 1<br><br>
> What if we used mean instead of sum
> Doesn't change. Rescale the whole loss function by a constant doesn't really matter. We don't care the true value of the loss. <br><br>
> What if we used \
> <img width="296" alt="스크린샷 2021-04-04 오후 7 25 27" src="https://user-images.githubusercontent.com/67621291/113505697-8502c780-957b-11eb-9539-22b86cd81a36.png">\
> Different. \
> Things that are very bad are now going to be squared bad. (Makes bad things worse). Quantify how much we care about different categories of errors.

<br>

``` python
def L_i_vectorized(x, y, W):
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + 1)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
 ```
 <br><br>
 `Q)` Suppose that we found a W such that L = 0. Is this W unique?\
 `A)` NO. 2W is also has L = 0\
 **At example - car**\
 <img width="200" alt="스크린샷 2021-04-04 오후 7 49 09" src="https://user-images.githubusercontent.com/67621291/113506294-d496c280-957e-11eb-9a5b-1554e216d316.png">
 <br><br>
 How is the classifier to choose btw these different versions of W that all achieve zero loss?
 → Really in practice, we don't actually care that much about fitting the training data. \We really care about the performance of this classifier on test data.\
 <img width="334" alt="스크린샷 2021-04-04 오후 7 54 16" src="https://user-images.githubusercontent.com/67621291/113506384-8afaa780-957f-11eb-8110-488b92aee010.png">\
 > ● is training data, ■ is new data. Green line is prefered.
 
 <br><br>
 `We usually solve it using concept of regularization`\
 <img width="500" alt="스크린샷 2021-04-04 오후 7 58 17" src="https://user-images.githubusercontent.com/67621291/113506477-1b38ec80-9580-11eb-82a6-b289952ae14e.png">\
**Occam's Razor:** Among competing hypotheses, the simpler is the best\
Hyperparameter lambda trades off between the two
<br><br>

---
### Regularization
In common use:\
**L2 regularization** : penalizing the euclidean norm of weigh vector\
L1 regularization, Elastic net(L1 + L2), Max norm regularization, \
(more specific to Deep learning) Dropout, Fancier : Batch normalization, stochastic depth
<br><br>
`L1 regularization` measure model complexity by the number of zeros in the weigh vector.\
It generally prefers sparse solution, that it drives all your entries of W to zero for most of the entries, except for a couple where it's allowed 
to deviate from zero.\
`L2 regularization` spread the W across all the values are less complex.<br><br>

---
### Loss function #2 - Softmax Classifier (Multinomial Logistic Regression)
<img width="200" alt="스크린샷 2021-04-04 오후 9 07 17" src="https://user-images.githubusercontent.com/67621291/113508226-bedaca80-9589-11eb-93a4-822a32504a94.png"><br><br> 

**Example**\
<img width="700" alt="스크린샷 2021-04-04 오후 9 09 30" src="https://user-images.githubusercontent.com/67621291/113508274-0e20fb00-958a-11eb-87f0-f88f4d69d262.png">
> Min possible loss *L_i* is 0 (-log(1))\
> Max possible loss *L_i* is infinity (-log(0))<br><br>
> Usually at initialization W is small so all s = 0. What is the loss?/
> log(C) (use this thing in debugging at first iteration)

<br>

**Multiclass SVM loss vs Softmax**\
If car score is much better than all the incorrect classes, jiggling the scores for that car image didn't change the multiclass SVM loss.\
Because the only thing that the SVM loss cared about was getting that correct score to be greater than a margin above the incorrect scores.\
But Softmax loss always wants to drive that probability mass all the way to one.\
So even if you're giving very high score to the correct class, and very low score to all the incorrect classes, softmax will want you to pile more and more 
probability mass on the correct class, and continue to push the score of that correct class up towards infinity, and the score of the incorrect classes 
down towards -infinity.\
`➣ SVM get data point over the bar to be correctly classified, and SOftmax always try to continually improve every single data point to get better.` \
`(But these two perform similarly)`<br><br>

---

### Optimization
: try to find the W that minimizes final loss function<br><br>
**Strategy #1 : A first very bad idea solution: Random Search**<br><br>
**Strategy #2 : Follow the slope**\
In multiple dimensions, the *gradient* is the vector of (partial derivatives) along each dimension.\
The slope in any direction is the *dot product* of the direction with the gradient.\
The direction of steepest descent is the *negative gradient.*<br>
- Numerical gradient : approximate, slow, easy to write
- Analytic gradient : exact, fast, error-prone

✔︎ In practice: Always use analytic gradient, but check implementation with numerical gradient.(debugging strategy) This is called a *gradient check*
``` python
# Vanila Gradient Descent
# First, initialize W randomly
while True:
  weights_grad = evaluate_gradient(loss_func, data, weights)
  weights += -step_size * weights_grad  # perform parameter update in the opposite of the gradient direction
```
Gradient was poining in the direction of greatest increase of the function, so - gradient points in the direction of greatest decrease.\
Step size is hyperparameter. Step size sometimes called a *learning rate*\
There exist different update rules <br><br><br>
**Stochastic Gradient Descent (SGD)**\
N(# of training set) is very large. So, computing this loss could be very expensive.\
And because the gradient is linear operator, when you actually try to compute the gradient, gradient of out loss is now the sum of the gradient of the losses
for each of the individual terms. → super slow\
✔︎ We use SGD<br><br>
`Stochastic Gradient Descent (SGD)`\
: at every iteration, we sample some small set of training examples, called a minibatch.  
32 / 64 / 128 common \
We'll use this small minibatch to compute an estimate of the full sum, and an estimate of the true gradient.
``` python
# Vanilla Minibatch Gradient Descent
while True:
  data_batch = sample_training_data(data, 256)                      # sample 256 examples
  weights_grad = evaluate_gradient(loss_func, data_batch, weights)
  weights += - step_size * weights_grad                             # perform parameter update
```

---

### Aside: Image Features
In practice, feeding raw pixel values into linear classifiers tends to not work so well.\
Before the dominance of deep neural networks, common was to have two-stage approach.<br><br>
Take image and compute various feature representations of that image\
Computing different kinds of quantities relating to the appearance of the image, and then concatenate these different feature vectors\
↓\
Feature representation of the image would be fed into a linear classifier<br><br>
<img width="550" alt="스크린샷 2021-04-04 오후 10 58 21" src="https://user-images.githubusercontent.com/67621291/113511136-44b24200-9599-11eb-9ce1-ecfd5d20692e.png">
<br><br>
**Feature Representation**<br>
<img width="500" alt="스크린샷 2021-04-04 오후 11 04 05" src="https://user-images.githubusercontent.com/67621291/113511282-0ff2ba80-959a-11eb-880a-81caa1704669.png">
<img width="500" alt="스크린샷 2021-04-04 오후 11 04 37" src="https://user-images.githubusercontent.com/67621291/113511300-226cf400-959a-11eb-8930-f28c329b9b6f.png">
<br><br>
<img width="600" alt="스크린샷 2021-04-04 오후 11 16 30" src="https://user-images.githubusercontent.com/67621291/113511649-cdca7880-959b-11eb-8cc8-57511bb2b72c.png">
▷ Inspiring from NLP

<br><br>
<img width="600" alt="스크린샷 2021-04-04 오후 11 26 11" src="https://user-images.githubusercontent.com/67621291/113511903-277f7280-959d-11eb-8300-47c6e7ef1e15.png">\
Fearue extractor would be fixed block. That would not be update during training.\
And during training, you only update the linear classifier if it's working on top of features.<br><br>
At ConvNet, we're going to learn the features directly from the data.\
So we'll take our raw pixels and feed them into ConvNet. And train entire weights for this entire network rather than just the weights of linear classifier on top.
