# 🌿 Bean Disease Classifier: Comparative Performance Analysis
##  Project Overview
As part of the harvard Foundations of Tiny ML we were tasked with an assignment to create a bean classifier. In this project, I developed a convolutional neural network with collaboration with a dense network to classify the health of bean plant leaves. Using a dataset of 224x224 pixel color images from Uganda, the goal was to accurately distinguish between **healthy** leaves and two specific diseases: **Bean Rust** and **Angular Leaf Spot**.



## Experimental Setup
To find the most effective model, I experimented with four different model architectural designs each version changed the transition from the convolutional base to the final classification layers:

| Version | Feature Transition Layer | Regularization |
| :--- | :--- | :--- |
| **Version 1** | `GlobalAveragePooling2D` | Dropout (0.2) |
| **Version 2** | `GlobalAveragePooling2D` | None |
| **Version 3** | `Flatten` | Dropout (0.2) |
| **Version 4** | **`Flatten`** | **None (Best Performer)** |

---

##  Performance Results
Below are the performance charts for each model version.
### Training vs. Validation Accuracy

*VERSION 1 PERFOMANCE*

<img width="580" height="422" alt="Image" src="https://github.com/user-attachments/assets/e2ed81a7-c64c-4fff-ae83-b76cbf0ba05e" />

*VERSION 2 PERFOMANCE*

<img width="566" height="441" alt="Image" src="https://github.com/user-attachments/assets/19d7d2a4-6d2d-44fa-baec-ec6dfa5546d0" />

*VERSION 3 PERFOMANCE*

<img width="587" height="456" alt="Image" src="https://github.com/user-attachments/assets/412e2f8e-a242-47ad-9837-177d24b30827" />

*VERSION 4 PERFOMANCE*

<img width="583" height="459" alt="Image" src="https://github.com/user-attachments/assets/ccb7a8b7-49bd-4537-8add-02116a670597" />


> **Key Finding:** Version 4 achieved the highest validation accuracy and the most stable convergence across all epochs.

---

##  Why Version 4 Performed Best
My analysis shows that the combination of a `Flatten` layer and the absence of `Dropout` created the ideal environment for this specific dataset.

### 1. Flattening vs. Global Pooling (Information Density)
The `GlobalAveragePooling2D` used in Versions 1 and 2 works by taking the average value of each feature map. While this reduces the number of parameters (making the model smaller), it also discards significant spatial information. 

In contrast, the `Flatten` layer in Version 4 preserves every pixel of the feature maps before passing them to the Dense layers. Since bean diseases like "Angular Leaf Spot" often appear as tiny, localized patterns, preserving this **spatial granularity** allowed the model to pick up on subtle visual cues that the pooling layers simply averaged away.

### 2. The Impact of Dropout (Capacity vs. Regularization)
Dropout is typically used to prevent overfitting by randomly "killing" neurons during training. However, in this project:
* **Over-regularization:** In Versions 1 and 3, Dropout likely removed too much information. Since the model wasn't drastically overfitting to begin with, Dropout acted as a "bottleneck" that prevented the model from learning the complex features of the leaves.
* **Full Capacity:** By removing Dropout in Version 4, I allowed the model to utilize its full learning capacity. This led to a more stable training signal and a faster path to high accuracy.


### 3. Architecture Synergy
By using `Flatten` (maximizing input data) and removing `Dropout` (maximizing processing power), Version 4 achieved the best balance of **expressive power**. The model had enough "memory" and detail to accurately categorize the three classes without being hindered by artificial noise.

---

## Final Conclusions
For this specific bean dataset, **detail preservation was more important than parameter reduction.** While Global Pooling and Dropout are industry standards for massive datasets (like ImageNet), for a specialized task like agricultural disease detection, allowing the model to see the "full picture" via flattening and full-neuron connectivity proved to be the winning strategy.

---



