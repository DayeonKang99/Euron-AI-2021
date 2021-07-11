# Visualizing and Understanding
What's going on inside ConvNets?<br><br>

## Visualize Filters 
**First layer**<br>
<img width="739" alt="스크린샷 2021-07-11 오후 3 32 21" src="https://user-images.githubusercontent.com/67621291/125185034-32f9ce00-e25d-11eb-844b-0d1fa3ad4d73.png"><br>
convolution layer. visualize weights <br><br>
<img width="550" alt="스크린샷 2021-07-11 오후 3 47 47" src="https://user-images.githubusercontent.com/67621291/125185344-59b90400-e25f-11eb-817f-d8f636915c00.png"><br><br>
**Last layer**<br>
4096-dimensional feature vector for an image (layer immediately before the classifier)<br> Run the network on many images, collect the feature vectors. <br>
- Nearest Neighbors: pixel로 하면 다른 이미지도 같아 pixel 수준에서 같아 보일 수 있다. feature로 Nearest Neighbor를 하게 되면 아래 사진처럼 코끼리가 다른 위치에 있어도 유사한 사진으로 판별 가능하다<br><img width="350" alt="스크린샷 2021-07-11 오후 3 56 23" src="https://user-images.githubusercontent.com/67621291/125185570-8d485e00-e260-11eb-9c1b-f48bbafdf46d.png">
- Dimensionality Reduction: visualize the "space" of FC7 feature vectors by reducing dimensionality of vectors from 4096 to 2 dimensions.
  * Simple algorithm: Principle Component Analysis (PCA)
  * More complex: t-SNE<br><img width="500" alt="스크린샷 2021-07-11 오후 4 05 43" src="https://user-images.githubusercontent.com/67621291/125185814-db119600-e261-11eb-9039-7eb4db2d17db.png">
<br><br><br>

visualize intermediate feature
## Visualizing Activations
the green box, this particular slice of the feature map of this layer of this particular network is maybe looking for human faces.<br>
<img width="350" alt="스크린샷 2021-07-11 오후 4 13 20" src="https://user-images.githubusercontent.com/67621291/125186023-eaddaa00-e262-11eb-8fac-66f4a7c4cb98.png"><br><br><br>

visualize intermediate feature
## Maximally Activating Patches 
Pick a layer and a channel; e.g. conv5 is 128x13x13, pick channel 17/128<br>
Run many images through the network, record values of chosen channel. <br>Visualize imgae patches that correspond to maximal activation.<br>
<img width="367" alt="스크린샷 2021-07-11 오후 6 24 47" src="https://user-images.githubusercontent.com/67621291/125189879-49138880-e275-11eb-84bb-b70f616195ab.png"><br><br>

## Occlusion Experiments 
Mask part of the image before feeding to CNN, draw heatmap of probability at each mask location<br>
일부를 가렸을 때 scorer가 급격히 변하면 그 부분이 classification decision에 중요<br>
<img width="884" alt="스크린샷 2021-07-11 오후 7 07 58" src="https://user-images.githubusercontent.com/67621291/125191027-529fef00-e27b-11eb-8a13-315213aa1d1f.png"><br><br>

## Saliency Maps 
How to tell which pixels matter for classification?<br>
Computer gradient of (unnormalized) class score with respect to image pixels, take absolute value and max over RGB channels <br>
<img width="513" alt="스크린샷 2021-07-11 오후 7 11 14" src="https://user-images.githubusercontent.com/67621291/125191126-c5a96580-e27b-11eb-939c-5d57662e4f55.png"><br><br>

## Intermediate Features via (guided) backprop
Pick a single intermediate neuron, e.g. one value in 128x13x13 conv5 feature map<br>Computer gradient of neuron value with respect to image pixels<br>
Image come out nicer if you only backprop positive gradients through each ReLU (guided backprop)<br>
<img width="589" alt="스크린샷 2021-07-11 오후 7 17 45" src="https://user-images.githubusercontent.com/67621291/125191311-af4fd980-e27c-11eb-9ee3-9e162ac9b78d.png"><br><br>

## Visualizing CNN features: Gradient Ascent
**(Guided) backprop:** Find the part of an image that a neuron responds to <br>
**Gradient ascent:** Generate a synthetic image that maximally activates a neuron<br>
<img width="300" alt="스크린샷 2021-07-11 오후 8 13 18" src="https://user-images.githubusercontent.com/67621291/125192770-7156b380-e284-11eb-8ef4-c33c3f6bae4f.png"><br><br>
<img width="700" alt="스크린샷 2021-07-11 오후 8 15 06" src="https://user-images.githubusercontent.com/67621291/125192834-b11d9b00-e284-11eb-8d79-32fe4c9cf3d1.png"><br><br>
<img width="200" alt="스크린샷 2021-07-11 오후 8 20 50" src="https://user-images.githubusercontent.com/67621291/125193048-7ec06d80-e285-11eb-8398-8c76c91670af.png"><br>
`Simpler regularizer`: Penalize L2 norm of generated image<br>
`Better regularizer`: Penalize L2 norm of image; also during optimization periodically<br>
1. Gaussian blur image
2. Clip pixels with small values to 0
3. Clip pixels with small gradients to 0

Use the same approach to visualize intermediate features<br>
<img width="700" alt="스크린샷 2021-07-11 오후 8 26 41" src="https://user-images.githubusercontent.com/67621291/125193229-4f5e3080-e286-11eb-855f-869d567105be.png"><br><br>
`Adding "multi-faceted" visualization` gives even nicer results:<br>(Plus more careful regularization, center-bias)
<img width="800" alt="스크린샷 2021-07-11 오후 8 53 37" src="https://user-images.githubusercontent.com/67621291/125193954-145dfc00-e28a-11eb-947e-82a3e9bffd15.png"><br><br>

## Fooling Images 
1. Start from an arbitrary image
2. Pick an arbitrary class
3. Modify the image to maximize the class
4. Repeat until network is fooled 

<img width="500" alt="스크린샷 2021-07-11 오후 8 59 02" src="https://user-images.githubusercontent.com/67621291/125194113-d3b2b280-e28a-11eb-9143-1d4839a6f45e.png"><br>

## DeepDream: Amplify existing features
Rather than synthesizing an image to maximize a specific neuron, instead try to amplify the neuron activations at some layer in the network<br><br>
Choose an image and a layer in a CNN; repeat:
1. Forward: compute activations at chosen layer
2. Set gradient of chosen layer *equal* to its activation
3. Backward: compute gradient on image
4. Update image
<br>

<img width="700" alt="스크린샷 2021-07-11 오후 9 09 38" src="https://user-images.githubusercontent.com/67621291/125194442-4ff9c580-e28c-11eb-9104-2e15e0b4bb9f.png"><br><br>

<img width="500" alt="스크린샷 2021-07-11 오후 9 14 30" src="https://user-images.githubusercontent.com/67621291/125194608-fcd44280-e28c-11eb-841b-c458841bb1da.png"><br><br>

## Feature Inversion
Given a CNN feature vector for an image, find a new image that:
- Matches the given feature vector
- "looks natural" (image prior regularization)

<img width="700" alt="스크린샷 2021-07-11 오후 9 18 44" src="https://user-images.githubusercontent.com/67621291/125194760-94d22c00-e28d-11eb-8f20-93b0360677b8.png"><br><br>
Reconstructing from different layers of VGG-16<br>
<img width="800" alt="스크린샷 2021-07-11 오후 9 19 43" src="https://user-images.githubusercontent.com/67621291/125194797-b7644500-e28d-11eb-86b6-bf8c5a069b84.png"><br>
network의 deeper part에서 relu4_3, relu5_1, relu5_3으로부터 reconstruct하면 low level details에 대한 information이 network의 higher layer에서 loss되어서 원래의 이미지와 비슷하게 보이지 않는다
<br><br>

## Texture Synthesis
Given a sample patch of some texture, can we generate a bigger image of the same texture?<br><br>
### Nearest Neighbor 
no neural networks here<br>
Generate pixels one at a time in scanline order; from neighborhood of already generated pixels and copy nearest neighbor from input <br>
<img width="600" alt="스크린샷 2021-07-11 오후 9 29 13" src="https://user-images.githubusercontent.com/67621291/125195080-0b235e00-e28f-11eb-9d8b-915b46edab53.png"><br><br>
### Neural Texture Synthesis: Gram Matrix 
<img width="800" alt="스크린샷 2021-07-11 오후 9 33 42" src="https://user-images.githubusercontent.com/67621291/125195206-ab798280-e28f-11eb-8488-6f3d7022cf26.png"><br>
Average over all HW pairs of vectors, giving **Gram matrix** of shape C x C<br>
Efficient to compute; reshape features from <br>
C x H x W to = C x HW<br>
then compute G = F F.T<br><br>

<img width="850" alt="스크린샷 2021-07-11 오후 9 38 37" src="https://user-images.githubusercontent.com/67621291/125195395-5b4ef000-e290-11eb-929c-447c61bfbeba.png"><br><br>
Gram matrix matching으로 texture synthesis.<br> 각각 pretrained convolution network의 다른 layer에서 gram matrix를 사용했을 때 결과<br>
Reconstructing texture from higher layers recovers larger features from the input texture<br>
<img width="300" alt="스크린샷 2021-07-11 오후 9 44 23" src="https://user-images.githubusercontent.com/67621291/125195569-298a5900-e291-11eb-9382-391a527e4475.png"><br><br>

## Neural Style Transfer 
Feature + Gram Reconstruction<br>
<img width="700" alt="스크린샷 2021-07-11 오후 9 58 04" src="https://user-images.githubusercontent.com/67621291/125195984-137d9800-e293-11eb-8c15-5674c65863f8.png"><br><br>
<img width="800" alt="스크린샷 2021-07-11 오후 9 58 35" src="https://user-images.githubusercontent.com/67621291/125196003-255f3b00-e293-11eb-97f3-b412ed899f64.png"><br><br>

Resizing style image before running style transfer algorithm can transfer different types of features<br>
<img width="600" alt="스크린샷 2021-07-11 오후 10 01 55" src="https://user-images.githubusercontent.com/67621291/125196136-9dc5fc00-e293-11eb-8320-4a203b03214e.png">

**Multiple Style Images**<br>
Mix style from multiple images by taking a weighted average of Gram matrices <br>
<img width="452" alt="스크린샷 2021-07-11 오후 10 05 06" src="https://user-images.githubusercontent.com/67621291/125196238-0f05af00-e294-11eb-99bc-3e0030118041.png"><br><br>

**Problem of Neural Style Transfer:**<br>Style transfer requires many forward / backward passes through VGG; very SLOW!<br><br>
**Solution:**<br>Train another neural network to perform style transfer for us<br><br>
### Fast Style Transfer
1. Train a feedforward network for each style 
2. Use pretrained CNN to compute same losses as before
3. After training, stylize images using a single forward pass

<img width="700" alt="스크린샷 2021-07-11 오후 10 13 51" src="https://user-images.githubusercontent.com/67621291/125196522-4759bd00-e295-11eb-9f19-80f104e876ca.png"><br><br>

another algorithm<br>
<img width="500" alt="스크린샷 2021-07-11 오후 10 15 43" src="https://user-images.githubusercontent.com/67621291/125196624-8a1b9500-e295-11eb-88aa-9a6bfb9c8091.png"><br>
Replacing batch normalization with Instance Normalization improves results<br><br>
### One Network, Many Styles
<img width="750" alt="스크린샷 2021-07-11 오후 10 18 26" src="https://user-images.githubusercontent.com/67621291/125196748-eb436880-e295-11eb-89b4-e0e6d37a34d8.png"><br><br>
---
### Summary
Many methods for understanding CNN representations<br><br>
**Activations:** Nearest neighbors, Dimensionality reduction, Maximal patches, Occlusion<br>
**Gradients:** Saliency maps, Class visualization, Fooling images, Feature inversion<br>
**Fun:** DeepDream, Style transfer
