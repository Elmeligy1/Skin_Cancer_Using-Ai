# Skin Cancer Classification Using AI

## Overview
This project demonstrates the application of Artificial Intelligence (AI) for classifying seven types of skin lesions from dermoscopic images. Utilizing a Convolutional Neural Network (CNN) trained on the HAM10000 dataset, the model achieves a test accuracy of **78.2%**, contributing to the automation and enhancement of skin cancer diagnosis.

## Features
- **Dataset**: HAM10000, a collection of 10,015 dermoscopic images labeled by expert consensus or histopathological diagnosis.
- **Model Architecture**: Optimized CNN with layers for convolution, pooling, and dropout to extract and classify lesion features.
- **Preprocessing**: Techniques including resizing, normalization, data augmentation, and missing data handling.
- **Visualization**: Performance tracking through accuracy/loss curves and confusion matrices.
- **Applications**: Aids dermatologists by providing a non-invasive diagnostic tool to classify lesions.

### Results
- **Test Accuracy**: 76.03%
- **Validation Accuracy**: 72.6%

## Challenges and Recommendations
- **Challenges**:
  - Class imbalance in datasets.
  - Overfitting during training.
  - Model generalization to unseen data.
- **Recommendations**:
  - Experiment with hyperparameter tuning.
  - Explore ensemble methods for accuracy improvements.
  - Deploy the model as a mobile or web application for clinical use.

## References
1. Esteva, A. et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*.
2. Haenssle, H. et al. (2018). Man against machine: diagnostic performance of a deep learning CNN. *Annals of Oncology*.
3. HAM10000 Dataset: [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).
