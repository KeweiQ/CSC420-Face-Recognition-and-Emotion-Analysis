# Video Demo

## Introduction & Member Intro

## The Problem
### Facial Expression Analysis
- Analysis facial expressions and connect them to known emotional states
- Enable the capability with any image with a face contained

### Overall Goal
- Read Slides

## Previous Accomplishments
- Read slides or
- Refer to project proposal

## Our Algorithm
Inspired by and modified from Eigenfaces
- Use the model to predict facial expressions instead of faces

### (Added) Viola-Jones Object Detection Framework
- To enable "versatility": capability to work on general photos
- Paul Viola and Micharl Jones, 2001
- Haar-like Features, Cascaded Classifier
- Read slides

### (Replaced) Fisherfaces
- A combination of Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA)
- Finds a linear combination of features with the consideration of classes, so discriminative information will not be lost

### Neural Network
- CNN

## Proposed Experiments
### Capability Justification

### Facial Extraction Algorithm Comparison
How much benefit are we receiving by using Fisherfaces instead of Eigenfaces? 
- Run the experiment respectively with Fisherfaces and Eigenfaces in the feature extraction step
- Compare the accuracy, precision, recall, etc. to see the benefits in performance obtained by using Fisherfaces

