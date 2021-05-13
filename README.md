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

#### Training Data Composition:

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



## Requirements:

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

