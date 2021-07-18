# Generative Models 
## Supervised vs Unsupervised Learning
### Supervised Learning 
Data: (x, y) → x is data, y is label<br>Goal: Learn a function to map x → y<br>
Examples: Classification, regression, object detection, semantic segmentation, image captioning, etc<br><br>
### Unsupervised Learning
Data: x → just data, no labels! (Training data is cheap) <br>Goal: Learn some underlying hidden *structure* of the data<br>
Examples: Clustering, dimensionality reduction, feature learning, density estimation, etc<br><br><br>

## Generative Models 
Given training data, generate new samples from same distribution<br>
<img src="https://user-images.githubusercontent.com/67621291/126063850-2a28cdea-87f5-4581-bfe8-fbaefa144a9a.png" width="630"><br><br>

Why Generative Mdoels?<br>
- Realistic samples for artwork, super-resolution, colorization, etc.
- Generative modesl of time-series data can be used for simulation and planning (reinforcement learning applications!)
- Training generative models can also enable inference of latent representations that can be useful as general features 
<br><br>
<img src="https://user-images.githubusercontent.com/67621291/126064206-e50f591d-1549-40c0-b407-5cc3644e6e9d.png" width="650"><br><br><br>

## PixelRNN and PixelCNN
**Explicit density model**<br>Use chain rule to decompose likelihood of an image x into product of 1-d distributions:<br>
<img src="https://user-images.githubusercontent.com/67621291/126064287-862d1dfb-d164-4e67-a6ee-c4e44bc3c059.png" width="300"><br>
Then maximize likelihood of training data <br>
Complex distribution over pixel values → Express using a neural network! Will need to define ordering of "previous pixels"<br><br>

### PixelRNN
Generate image pixels starting from corner<br>Dependency on previous pixels modeled using an RNN(LSTM)<br>
<img src="https://user-images.githubusercontent.com/67621291/126064459-082c5873-3ec2-4a92-a623-268fc238173f.png" width="150"><br>
**Drawback**: sequential generation is slow!<br><br>

### PixelCNN
Still generate image pixels starting from corner <br>Dependency on previous pixels now modeled using a CNN over context region <br>
Training: maximize likelihood of training images<br>
<img src="https://user-images.githubusercontent.com/67621291/126064537-3ff82734-12af-4667-8858-34373a829461.png" width="230"><br>
classification label is 0~255 (pixel values, using input data to create our loss)<br><br>
Training is faster than Pixel RNN (can parallelize convolutions since context region values known from training images)<br>
Generation must still proceed sequentially → Still SLOW<br><br>

### Pros and Cons
Pros:
- can explicitly compute likelihood p(x)
- explicit likelihood of training data gives good evaluation metric
- good samples

Cons:
- sequential generation → slow
<br><br><br>

## Variational Autoencoders (VAE)
VAEs define intractable density function with latent **z**:<br>
<img src="https://user-images.githubusercontent.com/67621291/126064962-553dc5cc-6e4e-4955-b313-bd862e63dedf.png" width="300"><br>
Cannot optimize directly, derive and optimize lower bound on likelihood instead<br><br>

### Some background first: Autoencoders
Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data<br>
Encoder is going to be a function mapping from input data to feature Z<br><br>
Encoder / Decoder is <br>
Originally: Linear + nonlinearity (sigmoid)<br>Later: Deep, fully-connected <br>Later: ReLU CNN<br><br>
**z** usually smaller than **x** (dimensionality reduction)<br>
`How to learn this feature representation?`<br>Train such that features can be used to reconstruct original data <br>"Autoencoding" - encoding itself<br><br>
<img src="https://user-images.githubusercontent.com/67621291/126065296-aee9a26f-a0f5-4460-b25e-227b474d3cdb.png" width="430">
<img src="https://user-images.githubusercontent.com/67621291/126065304-38b4aeca-66f4-4179-b078-619cf1f3d309.png" width="130"><br><br>
After training, throw away decoder<br>Encoder can be used to initialize a **supervised** model <br>
<img src="https://user-images.githubusercontent.com/67621291/126065425-c1ffc104-ce11-47e5-bdf7-71c4ae272be0.png" width="340"><br>
Train for final task (sometimes with small data)<br><br>

### Variational Autoencoders
Probabilistic spin on autoencoders - will let us sample from the model to generate data!
<img src="https://user-images.githubusercontent.com/67621291/126065541-c84f5a9e-b7fa-402b-beeb-1e1af2405fb6.png" width="530"><br>
**Intuition** (remember from autoencoders): <br>x is an image, z is latent factors used to generate x: attributes, orientation, etc<br><br>
`How should we represent this model?`<br>Choose prior p(z) to be simple, e.g. Gaussian. <br>Reasonable for latent attrubutes<br>
Conditional p(x|z) is complex (generates image) → represent with neural network <br><br>
`How to train the model?` <br>Learn model parameters to maximize likelihood of training data <br> ➡️ Problem is "Intractable"<br><br>
<img src="https://user-images.githubusercontent.com/67621291/126065855-dfa558f4-4fcc-47bc-8ffa-580e567a15aa.png" width="350"><br>
<img src="https://user-images.githubusercontent.com/67621291/126065960-78b169e8-15d0-4e90-aa63-a3d1ed7f3397.png" width="600"><br><br>
**Solution:**<br>
In addition to decoder network modeling p(x|z), define additional encoder network q(z|x) that approximates p(z|x)<br>
Will see that this allows us to derive a lower bound on the data likelihood that is tractable, which we can optimize <br><br>

<img src="https://user-images.githubusercontent.com/67621291/126066123-65153a03-c439-4d92-9103-e43da848f1e4.png" width="700"><br>
Encoder and decoder networks also called `"recognition" / "inference"` and `"generation"` networks <br><br>
Now equipped with our encoder and decoder netwokrs, let's work out the (log) data likelihood:<br>
<img src="https://user-images.githubusercontent.com/67621291/126068244-9e437f00-cdbb-4ee7-8479-0b658a182a1c.png" width="700"><br>
Dkl is KL divergence term (how close two distributions are)<br>
first two term is **Tractable lower bound** which we can take gradient of and optimize! (p(x|z) differentiable, KL term differentiable)<br><br>
<img src="https://user-images.githubusercontent.com/67621291/126068408-5d4073e5-5977-4906-ae56-795e854b01f7.png" width="700"><br><br>

**For every minibatch of input data: compute this forward pass, and then backprop!**<br>
get gradient, update model in order to maximize the likelihood of the trained data<br>
<img src="https://user-images.githubusercontent.com/67621291/126068653-6f96487d-6a78-4359-ab98-478f9b9139ba.png" width="700"> <br><br>

**Generating Data**<br>Use decoder network. Now sample z from prior!<br>
<img src="https://user-images.githubusercontent.com/67621291/126068835-8ecc850d-7211-4ea7-8b97-2925abae89eb.png" width="400">
<img src="https://user-images.githubusercontent.com/67621291/126068844-8c969574-4a4c-45d1-8090-b6b1a0264b25.png" width="230"><br>
Diagonal prior on **z** → independent latent variables <br>Different dimensions of **z** encode interpretable factors of variation <br>
e.g., z1 = degree of smile, z2 = head pose <br>
z variation is also good feature representation that can be computed using q(z|x)<br><br>

### Pros and Cons
**summary:**<br>Probabilistic spin to traditional autoencoders → allows generating data <br>Defines an intractable density → derive and optimize a (variational) lower bound<br><br>
**Pros:**
- Principled approach to generative models
- Allows inference of q(z|x), can be useful feature representation for other tasks

**Cons:**
- Maximizes lower bound of likelihood: okay, but not as good evaluation as PixelRNN/PixelCNN
- Samples blurrier and lower quality compared to state-of-the-art (GANs)
<br><br><br>

## Generative Adversarial Networks (GANs)
Don't work with any explicit density function! <br>Instead, take game-theoretic approach: learn to generate from training distribution through 2-player game<br>
Sample from a simple distribution, e.g. random noise. Learn transformation to training distribution<br>
<img src="https://user-images.githubusercontent.com/67621291/126069732-0ac5c211-67c1-4698-bdc1-ccf44a352c29.png" width="250"><br><br>

### Training GANs: Two-player game
**Generator network**: try to fool the discriminator by generating real-looking images<br>
**Discriminator network**: try to distinguish btw real and fake images<br>
<img src="https://user-images.githubusercontent.com/67621291/126069832-79546b89-0f14-433c-9e08-0d2814fd74fd.png" width="570"><br><br>
Train jointly in **minimax game**<br>
<img src="https://user-images.githubusercontent.com/67621291/126069947-3ad41995-a7d4-4bd2-a012-af2dcacd51f9.png" width="650"><br>
- Discriminator (theta d) wants to **maximize objective** such that D(x) is close to 1(real) and D(G(z)) is close to 0(fake)
- Generator (theta g) wants to **minimize objective** such that D(G(z)) is close to 1 (discriminator is fooled into thinking generated G(z) is real)
<br><br>

Alternate between:<br>
1. **Gradient ascent** on discriminator <br><img src="https://user-images.githubusercontent.com/67621291/126070426-e86acbb4-3667-4229-b885-922936e07e55.png" width="550">
2. **Gradient ascent*** on generator, different objective <br><img src="https://user-images.githubusercontent.com/67621291/126070469-0b515e51-9a77-47b8-bb08-25a9dab68915.png" width="300">

Instead of minimizing likelihood of discriminator being correct, now maximize likelihood of discriminator being wrong<br>
Same objective of fool discriminator, but now higher gradient signal for bad samples → works much better! Standard in practice<br>
<img src="https://user-images.githubusercontent.com/67621291/126070526-8c4b9501-74c9-4986-a367-fa4620ba6412.png" width="260"><br>
Aside: jointly training two networks is challenging, can be unstable. Choosing objectives with better loss landscapes helps training, is an active area of research.<br><br>

**Putting it together: GAN training algorithm**<br>
train discriminator for a bit basically <br><br>
<img src="https://user-images.githubusercontent.com/67621291/126070751-9c7ce32c-3496-4567-9e1b-95e81be0339f.png" width="600"><br>
`k steps` ▻ Some find k=1 more stable, others use k > 1, no best rule. <br>Recent work (e.g. Wasserstein GAN) alleviates this problem, better stability!<br><br>

### Generative Adversarial Nets: Convolutional Architectures
Generator is an upsampling network with fractionally-strided convolutions <br>Discriminator is a convolutional network<br>
<img src="https://user-images.githubusercontent.com/67621291/126070917-029eae5e-6e84-46c4-9d90-163a29d47a6d.png" width="650"><br><br>
<img src="https://user-images.githubusercontent.com/67621291/126070946-bb762c31-c3ca-4206-9dc7-8b49206de67a.png" width="500"><br><br>

### Generative Adversarial Nets: Interpretable Vector Math
<img src="https://user-images.githubusercontent.com/67621291/126071117-2f1c0c50-abd0-466d-b7f5-a13733606bb5.png" width="530"><br><br>

### Pros and Cons
**Pros:**
- Beautiful, state-of-the-art samples!

**Cons:**
- Trickier / more unstable to train
- Can't solve inference queries such as p(x), p(z|x)
