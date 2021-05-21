# Classifying Plant Health with YOLOv4 Detection

by Ashley Poon

## Problem Statement  

#### Can we classify whether a plant is healthy or unhealthy using object detection? 

The goal is to help novice plant owners determine if their plants are healthy or need some special attention. If the model is able to accurately classify an unhealthy plant, then users can take the next step to research and determine whether their plants are being over/under watered, recieving too much/too little sunlight, infested, etc. The guiding metric will be accuracy for CNN models and mean average precision for the YOLO model.

## Repo Contents
    
    
* 1.ImageProcessing_CNN.ipynb
    * Used to process crop and houseplant images for a basic CNN model
    
    
* 2.RedditScraping.ipynb
    * Used to scrape images from the subreddit r/plantclinic to collect additional images of unhealthy plants
    
    
* 3.ImageProcessing_CNN_withReddit.ipynb
    * Used to process crop, houseplant, and Reddit images for several CNN models    
    
    
* 4.Plant_Health_YOLOv4_Model.ipynb
    * Used to build and train a YOLO object detection model
    * This notebook is adapted from [The AI Guy](https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q)
    * The model was built and trained in Google Colab using the free GPU acceleration
    * Necessary files are located in the [yolov4](https://drive.google.com/drive/folders/14EO3jQPyI2OeHs5hLdtDXpN3hMY5izXA?usp=sharing) Gdrive folder
        * Folder content:
            * **backup**: partial and final training weights
            * **obj.zip**: training images and annotations (crop, houseplant, and Reddit)
            * **test.zip**: validation images and annotations (crop, houseplant, and Reddit)
            * **obj.names**: contains the class names
            * **obj.data**: contains specific info needed to set up model 
            * **yolov4-obj.cfg**: contains model configurations
            * **generate_train.py**: script to create a train.txt file that hold relative paths to all training images
            * **generate_test.py**: script to create a test.txt file that hold relative paths to all test images
            
* 5.Visualizations.ipynb
    * Used to create visualizations for CNN and YOLO model performance

#### [Click Here for the Executive Summary](ExecSummary.md)

## Image Data

#### Datasets


__[Crop Images](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset):

This kaggle dataset contains 88K lab images of healthy and diseased crop leaves. Crops included apple, corn, strawberry, tomato, etc. 

__[Houseplant Images](https://www.kaggle.com/russellchan/healthy-and-wilted-houseplant-images/version/1?select=houseplant_images):

This kaggle dataset contains 904 Google images of healthy and wilted houseplants. 

__[r/plantclinic](https://www.reddit.com/r/plantclinic/):

This data contains approximately 500 images scraped from the r/plantclinic subreddit. The majority of these images submitted by Reddit users are of unhealthy plants. Since these are plants in "real-life" environments, it should help the model train on images similar to what it is likely to detect in the future.

## Modeling Process 

### Cleaning & Preprocessing

**Data Cleaning**

- crop leaves: organized into healthy vs diseased (unhealthy) regardless of species 
- houseplants: organized into healthy vs wilted (unhealthy)
- r/plantclinic: each of the images were reviewed and classified as unhealthy or healthy/unknown

**Pre-processing for CNN**

- healthy plant dataset: sampled 500 healthy crop leaf images, 451 healthy houseplant images
- unhealthy plant dataset: sampled 500 unhealthy crop leaf images, 451 unhealthy houseplant images, and 314 Reddit images (for CNN models including Reddit data)
- used load_img to resize each image and img_to_array to convert each image to array

**Pre-processing for YOLO**

- healthy plant dataset: manually labeled 905 healthy plant images from the crop leaves and houseplant datasets using LabelImg
- unhealthy plant dataset: manually labeled 1028 unhealthy plant images from the crop leaves, houseplant, and Reddit datasets using LabelImg

### YOLO Training Data Composition

| Class                  | Source      | Num Images  
| ---                    | ---         | ---       
| Healthy                | Crop        | 457     
|                        | Houseplant  | 448  
| **Total Healthy**      |             | **905**
| Unhealthy              | Crop        | 278    
|                        | Houseplant  | 440    
|                        | Reddit      | 310
| **Total Unhealthy**    |             | **1028**
| **Total Images**       |             | **1933**

### Models

1) inital: CNN without Reddit images
2) optimized: CNN with Reddit images
3) final: YOLO4 object detection 

### YOLO Results

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

## Conclusion
Based on the mAP staying relatively flat with every thousand weights, it seems that the YOLO model has capped in terms of performance. Typically for YOLO models, mAPs increase significantly in the first couple thousand iterations and then plateaus in the next few thousand iterations. For this particular dataset, it makes sense that the model capped around low 80% given that it can sometimes be difficult to distinguish unhealthy plants from healthy ones, even for humans. The reason for this is because of all the different types of plants available, where a unhealthy characteristic of one plant might not be for another. An example of this is for some plants their leaves naturally grow downwards but it could be mistaken as wilted and therefore unhealthy by the model.

Similar to YOLO, the CNN model capped around 80% accuracy which is also likely due to the difficulty in classifying plant health for a wide variety of plants. While it seems that YOLO did not have a huge improvement over CNN, the benefit of YOLO is the ability to detect in real-time. For demonstration purposes, the YOLO 3000 weights iteration were used as it gives us high precision with additional training as opposed to the 1000 weights.

## Requirements

| Purpose                | Libraries  | Import Statements                                                                 
| ---                    | ---        | ---                                                                                
| General                | pandas     | import pandas as pd                                                               
|                        | numpy      | import numpy as np                                                                                                        
|                        | os         | import os                                                                         
| Webscraping            | requests   | import requests
| Modeling               | sklearn    | from sklearn.model_selection import train_test_split                       
|                        | tensorflow | import tensorflow as tf                                 
|                        |            | from tensorflow.keras.preprocessing.image import img_to_array, load_img                         
|                        |            | from tensorflow.keras.models import Sequential
|                        |            | from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
| Yolo                   | darknet    | !git clone https://github.com/AlexeyAB/darknet

