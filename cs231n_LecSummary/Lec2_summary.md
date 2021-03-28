# Image Classification pipeline
### Image Classification
: A core task in computer vision. Assigning one of fixed category labels to the image.<br>
The probelem is Semantic Gap.<br><br>
#### Challenges
- Viewpoint variation 
- Illumination
- Deformation
- Occlusion : You can only see part of a object.
- Background Clutter : The object could look similar in appearance to the background.
- Interclass variation : Object can come in different shapes, sizes, colors, and ages.<br>
#### Data-Driven Approach
1. Collect a dataset of images and labels
2. Use Machine Learning to train a classifier
3. Evaluate the classifier on new images
> ✔️ **API Function**
> - train : Input images and labels, and then output a model
> - predict : Input the model and then make prediction for images

Data-Driven Approach is much more general than just Deep Learning.<br><br>
***

### First calssifier: **Nearest Neighbor**
Training step : Memorize all data and labels\
Prediction step : Predict the label of the most similar training image<br>
> <img width="357" alt="스크린샷 2021-03-28 오후 5 39 27" src="https://user-images.githubusercontent.com/67621291/112746835-8f0d4f00-8fec-11eb-986b-256f5788e02a.png"><br>
> Using CIFAR10 dataset

It is not always correct, but looks quite visually similar.
<br><br>
**Compare test images and training images.**\
Distance Metric
→ L1 distance : Just compare individual pixels. <br><br>
```python
import numpy as np
class NearestNeighbor:
  def __init__(self):
    pass
    
  # Memorize training data
  def train(self, X, y):
    self.Xtr = X
    self.ytr = y
  
  def predict(self, X):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
  
    # For each test image: Find closest train image, Predict label of nearest image
    # Using the L1 distance
    for i in xrange(num_test):
      distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
      min_index = np.argmin(distances)
      Ypred[i] = self.ytr[min_indes]
     
    return Ypred
```
> N examples,\
> Train O(1)   (copying a pointer)\
> Predict O(N) (compare)\
> Prediction is SLOW
> We want classifiers that are FAST at prediction, SLOW for training is ok.

<br><br>
### K-Nearest Neighbors
: Instead of copying label from nearest neighbor, take majority vote from K closest points.\
K>1 tends to smooth out decision boundaries and lead to better result.\
<img width="878" alt="스크린샷 2021-03-28 오후 8 15 45" src="https://user-images.githubusercontent.com/67621291/112750231-62fcc880-9002-11eb-85d0-77b95c999e94.png"><br><br>
K-Nearest Neighbor on images **never used**
- Very slow at test time
- Distance metrics on pixels are not informative - L2 distance is not good at capturing perceptional distances between images.
- Curse of dimensionality - dimensional growth makes exponential growth of points to cover the space<br><br>

**Distance Metric**\
L1 (Manhattan) distance, L2 (Euclidean) distance\
L1 depends on the coordinate system.\
<img width="500" alt="스크린샷 2021-03-28 오후 8 37 43" src="https://user-images.githubusercontent.com/67621291/112750770-75c4cc80-9005-11eb-9c8f-5bec88dd0b36.png">\
Different distance metrics make different assumptions about the underlying geometry or topology.\
If your input features, if the individual entries in your vector have some important meaning for your task, then L1 might be a more natural fit.\
Bur if it's just a generic vector in some space and you don't know what they actually mean, then maybe L2 is slightly more natural.<br><br>

***
### Hyperparameters
- Choices about the algorithm that we set rather than learn
- like K and the distance metric
- Very probelm-dependent
- Must try them all out and see what works best
<br><br>
#### Setting Hyperparameters
Idea #1 : Choose hyperparameters that work best on the data
> BAD. e.g., K=1 always works perfectly on training data.
Idea #2 : Split data into train and test, choose hyperparameters that work best on test data
> BAD. No idea to how algorithm will perform on new data. performance on this test set will no longer be representative of our performance of new, unseen data.
Idea #3 : Split data into `train`, `val`, and `test`; choose hyperparameters on val and evaluate on test
> BETTER. Train algorithm with many different choices of hyperparameters on the training set, evaluate on the validation set, and now pick the set of hyperparameters which performs best on the validation set. 
Idea $4 : Cross-Validation : Split data(without test data) into `folds`, try each fold as validation and average the results
> Useful for small datasets, but not used too frequently in deep learning

<br><br>
***
### Linear Classification
: Simple learning algorithm. This will become very important and help us build up to whole neural networks and whole convolutional networks. \
Neural Networks like Lego blocks. You can have different kinds of components of neural networks and stick these components together to build large different towers of
convolutional networks.\
Most basic building blocks in different types of deep learning applications is Linear Classifier.
<br><br>
#### Parametric Model
parametric model has two different components.\
<img width="600" alt="스크린샷 2021-03-28 오후 10 22 12" src="https://user-images.githubusercontent.com/67621291/112753779-0e624900-9014-11eb-8216-28d4b75142de.png">
> x for input data

In the K-Nearest Neighbor setup there was no parameters. Just whole training set used that at test time.\
But, Prametric Approach summarize our knowledge of the training data and stick all that knowledge into these parameters, W.\
At test time, we no longer need the actual training data. We only need these parameters, W, at test time. → more efficient and actually run on small devices.\
Whole story of deep learning is coming up with right structure for this function, F.
<br><br>
<img width="300" alt="스크린샷 2021-03-28 오후 10 43 20" src="https://user-images.githubusercontent.com/67621291/112754426-02c45180-9017-11eb-8edd-38eb7406abf2.png">\
Stretch input image out into a long column vector\
We need 10 categories.\
`b` bias gives data independent preferences for some classes over another. <br><br><br>
<img width="600" alt="스크린샷 2021-03-28 오후 10 45 53" src="https://user-images.githubusercontent.com/67621291/112754508-5d5dad80-9017-11eb-894d-7efdaf029eb1.png"><br>
**Problem** : Linear classifier is only learning one template for each class.<br><br>
#### High dimensional point of view
Each of images is like a point in high dimensional space. Linear Classifier is putting in linear decision boundaries to try to draw linear seperation between 
one category and the rest of the cotegories.<br>
<img width="400" alt="스크린샷 2021-03-28 오후 10 59 13" src="https://user-images.githubusercontent.com/67621291/112754896-3acc9400-9019-11eb-945e-e3e8057e4ce3.png"><br><br>
**Problem - Hard cases** : We can't draw single linear line to seperate the blue from the red.\
<img width="650" alt="스크린샷 2021-03-28 오후 11 04 44" src="https://user-images.githubusercontent.com/67621291/112755051-ff7e9500-9019-11eb-92a3-83d341ce4546.png">
