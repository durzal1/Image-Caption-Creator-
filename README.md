# Image-Caption-Creator-

Given an image, this model is able to create a pseudo-accurate caption. However, due to the limited data, training time, and GPU power the comments have incorrect grammar and sentence structure. Although these short comings exist, the results show that the model knows what is happening in the images. 

# Examples 
![image](https://github.com/durzal1/Image-Caption-Creator-/assets/67489054/faa01a40-d404-4a43-8d23-df84be1f8169)
![image](https://github.com/durzal1/Image-Caption-Creator-/assets/67489054/5c708efa-3b79-404c-a71e-96f5cfc14ca2)
![image](https://github.com/durzal1/Image-Caption-Creator-/assets/67489054/ebfb1792-b608-4d8d-92a3-22aa2b03fdfb)

Although at some times it is vastly different from the caption, it does give a descriptive acceptable caption. 

# Architecture 

First, the image went through a pre-trained resnet to extract the features. Then, it went through an LSTM model with attention in order to give bias to more important pixels. The total training time was 35 hours. It went through about 250 epochs total. Unfortanetly, the model is too big to store in github (Over 100 mb).
