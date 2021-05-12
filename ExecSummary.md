# Classifying Plant Health with YOLOv4 Detection

by Ashley Poon

## Executive Summary  

### Background

The goal of this project is to help new plant owners detect whether their plants are healthy or unhealthy.

Images were collected through Kaggle of healthy and unhealthy crop leaves and houseplants. Additional images were collected from r/plantclinic, a subreddit where users can submit images of their unhealthy plants to get diagnosed.

Initial models built and trained were CNN models to get a baseline of performance. The final model was a YOLOv4 model which is able to detect whether a plant is healthy or unhealthy in real-time with bounding boxes.

### Problem Statement 

Can we classify whether a plant is healthy or unhealthy using object detection?

The goal is to help novice plant owners determine if their plants are healthy or need some special attention. If the model is able to accurately classify an unhealthy plant, then users can take the next time to research and determine whether their plants are being over/under watered, recieving too much/too little sunlight, infested, etc. The guiding metric will be accuracy.

### Cleaning & Pre-processing

**Data Cleaning**

Three different datasets were collected: healthy and diseased crop leaves, healthy and wilted houseplants, and images from r/plantclinic.
- crop leaves: originally organized into 38 different classes of healthy and diseased crop leaves (apple, corn, strawberry, etc.), images were reorganized by healthy vs unhealthy regardless of species 
- houseplants: organized into healthy vs wilted, recategorized wilted as unhealthy
- r/plantclinic: each of the images were reviewed and classified as unhealthy or healthy/unknown

**Pre-processing for CNN**

1) For the healthy plant dataset, sampled 500 healthy crop leaf images and 451 houseplant images. 

2) For the unhealthy plant dataset, sampled 500 unhealthy crop leaf images, 451 houseplant images, and 314 Reddit images (for CNN models including Reddit data).

3) Pre-process images for CNN using load_img to resize image and img_to_array to convert image to array.

**Pre-processing for YOLO**

1) For the healthy plant dataset, manually labeled and annotated 905 healthy plant images from the crop leaves and houseplant datasets using LabelImg.

2) For the unhealthy plant dataset, manually labeled and annotated 1028 unhealthy plant images from the crop leaves, houseplant, and Reddit datasets using LabelImg.

3) Define helper functions to handle images in Colab: display, upload, and download.


### Modeling

#### Baseline:

For the initial CNN model, the images 50/50 healthy and unhealthy. Therefore the baseline accuracy was 50%.

#### Initial: CNN without Reddit images 

After image processing, an initial CNN model was built to classify whether plants were healthy or unhealthy. The CNN model had two convolutional layers with 64 filters and one dense layer with 64 nodes.

| Layer (type)                        | Output Shape          | Param #
|---                                  | ---                   | ---
| conv2d (Conv2D)                     | (None, 254, 254, 64)  | 1,792 
| max_pooling2d (MaxPooling2D)        | (None, 127, 127, 64)  | 0
| conv2d_1 (Conv2D)                   | (None, 125, 125, 64)  | 36,928 
| max_pooling2d_1 (MaxPooling2D)      | (None, 62, 62, 64)    | 0
| flatten (Flatten)                   | (None, 246,016)       | 0 
| dense (Dense)                       | (None, 64)            | 15,745,088 
| dense_1 (Dense)                     | (None, 1)             | 65 

The best score with minimal overfitting was in epoch 4 with a train accuracy of 78% and test accuracy of 73%.

#### Optimized: CNN with Reddit images 

Additional CNN models were trained with the Reddit images included. Four different models were built, however the best CNN model had four convolutional layers with 16 filters and one dense layer with 64 nodes.

| Model  | Convolutional layers | Dense layers  | Best train accuracy | Best test accuracy
|---     | ---                  | ---           | ---                 | ---
| 1      | 4 (16 filters)       | 1 (16 nodes)  | 79%                 | 79% 
| 2      | 4 (16 filters)       | 1 (64 nodes)  | 82%                 | 80% 
| 3      | 4 (16 filters)       | 1 (128 nodes) | 85%                 | 80% 
| 4      | 3 (32 filters)       | 1 (32 nodes)  | 84%                 | 80% 

The best score with minimal overfitting was from model 2 with a train accuracy of 82% and test accuracy of 80%. With additional nodes in the dense layer, the model becomes increasingly overfit. Increasing the filters also has a similar effect of overfitting the model with no improvement in test accuracy.

#### Final: YOLOv4 object detection

The ultimate goal was to create a model that can be used for real-time detection to classify healthy or unhealthy plants. The YOLO model was trained on a custom dataset of 905 healthy plant images and 1028 unhealthy plant images (53% baseline accuracy). 

The parameters used for the model were:

- batch = 64
- subdivisions = 16
- max_batches = 6000
- steps = 4800, 5400
- classes = 2
- filters = 21

Below is a performance summary showing the mAP (mean average precision) of every 1000 weights, as well as the AP (average precision) of each class, precision, recall, and F1-score.

| Weights  | mAP    | healthy AP | unhealthy AP | Precision | Recall | F1-score
|---       |---     |---         |---           |---        |---     |---
| 1000     |84.01%  |81.57%      | 86.46%       |83%        |80%     |81%
| 2000     |82.60%  |81.01%      | 84.19%       |83%        |80%     |81% 
| 3000     |83.35%  |81.55%      | 85.14%       |86%        |82%     |84%
| 4000     |82.73%  |81.51%      | 83.94%       |88%        |82%     |85%
| 5000     |81.60%  |80.70%      | 82.49%       |86%        |78%     |82%
| 6000     |80.24%  |79.45%      | 81.04%       |84%        |78%     |81%

The 1000 weights had the highest mAP followed by 3000 weights, however the mAPs do not change much with every thousand weights. Every thousand weights had higher AP on the unhealthy class, minimized false negatives, and optimized for precision.

### Summary

Based on the mAP staying relatively flat with every thousand weights, it seems that the YOLO model has capped in terms of performance. Typically for YOLO models, mAPs increase significantly in the first couple thousand iterations and then plateaus in the next few thousand iterations. For this particular dataset, it makes sense that the model capped around low 80% given that it can sometimes be difficult to distinguish unhealthy plants from healthy ones, even for humans. The reason for this is because of all the different types of plants available, where a unhealthy characteristic of one plant might not be for another. An example of this is for some plants their leaves naturally grow downwards but it could be mistaken as wilted and therefore unhealthy by the model.

Similar to YOLO, the CNN model capped around 80% accuracy which is also likely due to the difficulty in classifying plant health for a wide variety of plants. While it seems that YOLO did not have a huge improvement over CNN, the benefit of YOLO is the ability to detect in real-time. For demonstration purposes, the YOLO 3000 weights iteration will be used as it gives us high precision with additional training as opposed to the 1000 weights.

### Recommendations

* __Immediate Use:__  Demo where users can upload an image and the model can predict whether the plant is healthy or not.
* __Next Steps:__  
    1. __Train on additional plant images:__  As mentioned, the model may confuse a normal characteristic of one plant to be unhealthy on another plant and vice versa. The easiest way to potentially reduce the confusion is to train the model on a wider variety and more images of healthy and unhealthy plants. The immediate next step could be to scrape more images from the r/plantclinic. Additionally, the "real-life" images available on the subreddit would help the model learn better on images that it can expect to receive in real-time.
    2. __Create mobile app:__  Once the model is optimized, it can be used to create a front-end app where users can use to help detect whether their plant is healthy or unhealthy. Users can either use their phone cameras to dectect plant health on the spot or set up a webcam and view detections on their phone while they are away. This can be used by people who travel often but still want check in on their plants, and also enables them to know when to ask someone to help care for any plant that needs special attention while they are away.