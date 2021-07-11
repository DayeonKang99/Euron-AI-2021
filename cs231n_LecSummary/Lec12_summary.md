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
<br><br>

## Visualizing Activations
the green box, this particular slice of the feature map of this layer of this particular network is maybe looking for human faces.<br>
<img width="509" alt="스크린샷 2021-07-11 오후 4 13 20" src="https://user-images.githubusercontent.com/67621291/125186023-eaddaa00-e262-11eb-8fac-66f4a7c4cb98.png">
