# Detection and Segmentation
<img width="800" alt="스크린샷 2021-07-09 오후 2 44 50" src="https://user-images.githubusercontent.com/67621291/125029175-386fe080-e0c4-11eb-9e7f-341554d0bb9b.png"><br>
<br><br>

## Semantic Segmentation
Lable each pixel in the image with a category label. Don't differentiate instances, only care about pixels.<br><br>
**Idea:** <br>
* **Sliding Window** (crop해서 classify) → `Problem)` very inefficient. Not reusing shared features btw overlapping patches <br>
* **Fully Convolutional:** design network as a bunch of convolutional layers to make predictions for pixels all at once → `Problem)` extremely computational expensive<br>
* **Fully Convolutional:** design network as a bunch of convolutional layers, with `downsampling` and `upsampling` inside the network. output image size == input image size. 모든 pixel에 대해 cross-entropy loss 사용해서 backprop <br>
> Downsampling: Pooling, strided convolution<br>
> Upsampling: Unpooling

<img width="800" alt="스크린샷 2021-07-10 오후 8 49 51" src="https://user-images.githubusercontent.com/67621291/125161966-61759b80-e1c0-11eb-81e9-6496922bc930.png"><br><br>

### Unpooling
<img width="852" alt="스크린샷 2021-07-10 오후 8 53 26" src="https://user-images.githubusercontent.com/67621291/125162058-e1036a80-e1c0-11eb-8160-b8350aa645aa.png"><br><br>
<img width="884" alt="스크린샷 2021-07-10 오후 8 55 37" src="https://user-images.githubusercontent.com/67621291/125162125-2fb10480-e1c1-11eb-96a9-5ef4b45c6951.png"><br>
corresponding pairs of downsampling and upsampling layers <br><br>
**Learnable Upsampling: Transpose Convolution**<br>
<img width="700" alt="스크린샷 2021-07-10 오후 9 12 04" src="https://user-images.githubusercontent.com/67621291/125162546-7c95da80-e1c3-11eb-9e48-d5778b016379.png"><br>
Other names: Deconvolution(bad), Upconvolution, Fractionally strided convolution, Backward strided convolution<br>
1D example:<br>
<img width="400" alt="스크린샷 2021-07-10 오후 9 21 38" src="https://user-images.githubusercontent.com/67621291/125162802-d2b74d80-e1c4-11eb-9e5e-026f72dc81a1.png"><br><br><br>

## Classification + Localization
<img width="800" alt="스크린샷 2021-07-10 오후 9 51 10" src="https://user-images.githubusercontent.com/67621291/125163619-f1b7de80-e1c8-11eb-8051-960b12b7400c.png"><br>
fully connected 이전 network는 often pretrained on ImageNet (transfer learning)<br><br>
**Aside:** Pose Estimation<br>
<img width="700" alt="스크린샷 2021-07-10 오후 9 59 41" src="https://user-images.githubusercontent.com/67621291/125163869-24160b80-e1ca-11eb-93bf-289359ed980d.png">
<img width="200" alt="스크린샷 2021-07-10 오후 10 00 36" src="https://user-images.githubusercontent.com/67621291/125163897-43ad3400-e1ca-11eb-957b-64fa8401be22.png">
<br><br>
`categorical output(classification) → cross entropy loss, softmax loss, SVM...`<br>
`continuous output (position of the points) → regression loss (L2, L1...)`<br><br><br>

## Object Detection
Unlike localization, each image needs a different number of outputs!<br><br>
- **Sliding Window:** apply a CNN to many different crops of the image, CNN classifies each crop as object or background → `Problem)` Need to apply CNN to huge number of locations and scales, very computationally expensive!
- **Region Proposals**
<br>

Region Proposals (not Deep learning)
- find "blobby" image regions that are likely to contain objects
- relatively fast to run; e.g. Selective Search gives 2000 region proposals in a few seconds on CPU
<br>

### R-CNN
<img width="750" alt="스크린샷 2021-07-10 오후 11 42 03" src="https://user-images.githubusercontent.com/67621291/125166831-70684800-e1d8-11eb-891f-9f38e7293973.png"><br>
**Problems:** 
- computationally expensive
- training is slow, takes a lot of disk space. 
- also slow in test time
<br>

### Fast R-CNN
<img width="850" alt="스크린샷 2021-07-10 오후 11 55 33" src="https://user-images.githubusercontent.com/67621291/125167200-52034c00-e1da-11eb-85c0-d6a6609e6f7d.png"><br>
at training) Multi-task loss (log loss + smooth L1 loss) and backprop through entire things<br>
**Problem:** Runtime dominated by region proposals<br><br>

### Faster R-CNN
<img width="700" alt="스크린샷 2021-07-11 오전 12 24 15" src="https://user-images.githubusercontent.com/67621291/125168016-55003b80-e1de-11eb-8a4e-7c2070eeaf9b.png"><br><br>

### Detection without Proposals: YOLO / SSD
<img width="900" alt="스크린샷 2021-07-11 오전 12 31 40" src="https://user-images.githubusercontent.com/67621291/125168228-5e3dd800-e1df-11eb-8d3f-38800fe8dce7.png"><br><br>
SSD is much faster but not as accurate. Faster R-CNN is slower but more accurate<br><br><br>

## Instance Segmentation
### Mask R-CNN
<img width="700" alt="스크린샷 2021-07-11 오전 12 44 02" src="https://user-images.githubusercontent.com/67621291/125168563-17e97880-e1e1-11eb-96c0-299de6d8551e.png">
