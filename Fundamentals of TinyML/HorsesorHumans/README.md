# Project Overview 

This project demonstrates how to perform computer vision and explores techniques such as data augmentation to improve the accuracy of a convolutional neural network using TensorFlow and Keras. The model classifies images from the Horse or Human dataset provided by Google.

# Characteristics of the Dataset

- The dataset contains two main categories training and validation files
- The training files contain 1027 images.
- The Validation contain 256 images.
- An overal total of 1283 images of non-consistent sizes and shapes.
- The images are all 3 dimensions meaning they have a height,width and number of channels(depth)
  
# Main Steps Involved 
## Data Loading

*The training dataset was loaded from the source namely:*
```
https://storage.googleapis.com/learning-datasets/horse-or-human.zip\O /tmp/horse-or-human.zip #downloads horse-or-human.zip and saves it to /tmp/horse-or-human.zip)
```
*The Validation dataset was loaded from the source namely:*
```
https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip
```


## Unzipping files

During loading the files were compressed in a zip file from the source this was so as to reduce loading time. After loading the files needed to be decompressed so as to return the imagesin their normal state to allow for processing. This was done through importing the python library zipfile which contains mechanism to open and extract the contents of the zipfile and load them into a normal directory(tmp) 

## Building Model
The model has 12 layers:

- Trainable layers: 7 layers (4 Conv2D + 3 Dense) 
- Pooling layers: 4 maxpool layers
- A single flattening layer

## Model Compilation

The model is configured for training using:

- The RMSprop - algorithm used to adjust parameters (learning rate = 0.0001)
- Binary crossentropy as the loss function, suitable for multi-class classification with integer labels.
- accuracy as the metric to monitor during training.

## Data Augmentation

Creating multiple variations of each training image by randomly rotating, shifting, zooming, and flipping them. This artificially expands the training dataset and teaches the model to recognize subjects regardless of their position, angle, or size in the frame.

## Creating Iterator
Applying ```flow_from_dierctory``` which is responsible for applying transformations defined under the image generator to the parents directory also resizes the images and does this in batch sizes of 128

## Model Training 
The model was trained on 100 epochs using the training iterator that was created before and also 8 steps per epoch meaning for a single rotaion of the model seeing the data, the data has to be fed to the model 8 times. This also gives a stopping point to the iterator which generates images continually.


## Visualization of Model Architecture
```
Input Image (100, 100, 3)
         │
         ▼
┌─────────────────────┐
│   Conv2D-1          │ 32 filters, 3x3, ReLU
│ Output: (98, 98, 32)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   MaxPooling2D-1    │ 2x2
│ Output: (49, 49, 32)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv2D-2          │ 64 filters, 3x3, ReLU
│ Output: (47, 47, 64)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   MaxPooling2D-2    │ 2x2
│ Output: (23, 23, 64)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv2D-3          │ 128 filters, 3x3, ReLU
│Output: (21, 21, 128)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   MaxPooling2D-3    │ 2x2
│Output: (10, 10, 128)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv2D-4          │ 256 filters, 3x3, ReLU
│  Output: (8, 8, 256)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   MaxPooling2D-4    │ 2x2
│  Output: (4, 4, 256)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Flatten        │
│   Output: 4096      │ (4*4*256)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Dense-1        │ 512 neurons, ReLU
│   Output: 512       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Dense-2        │ 256 neurons, ReLU
│   Output: 256       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Dense-3        │ 1 neuron, Sigmoid
│   Output: 1         │ (Binary classification)
└─────────────────────┘
         │
         ▼
    Prediction
  (0 or 1 - Horses/Humans)
```

# Model Evaluation

This 12-layer CNN architecture processes 100×100 RGB images for binary classification (horses versus humans). The model achieves 87.5% accuracy on training data but only 62.5% on validation data, revealing a significant 25% performance gap that indicates overfitting. While the model learns the training data well, it struggles to generalize to new, unseen images, with inconsistent validation performance peaking at 75.78%. 

# Diagnosis
The model's low performance can be attributed to insufficient steps per epoch. Since images are continuously shifted, rotated, and transformed during training, the model essentially sees a slightly different version of each image every epoch. This makes it harder to consistently learn core features, as the patterns keep changing. To boost performance, the model needs to see the data more times (more epochs) and be exposed to more examples overall, allowing it to gradually identify the key features that remain constant despite all the transformations

# Test the model 🎉

- Click the Colab Badge below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keden49/HarvardX-TinyML/blob/main/Fundamentals%20of%20TinyML/HorsesorHumans/HorsesOrHumans.ipynb)

- Click Runtime and then Run all to test out model

# Credits 
1. Freecodecamp - For providing the foundational Machine Learning with Python curriculum and project prompts
2. Google Colab: Provided the cloud-based interactive environment for development and easy model testing.
3. TensorFlow: The core machine learning platform used for building and training the deep neural network.
4. Keras: Used for its high-level API to define the model architecture and layers with simplicity and speed.
5. AI Collaborators (Gemini & DeepSeek): Assisted in breaking down complex concepts and greatly helped me to understand the workflow


