# Skin Cancer Classification Using AI

## Overview
This project demonstrates the application of Artificial Intelligence (AI) for classifying seven types of skin lesions from dermoscopic images. Utilizing a Convolutional Neural Network (CNN) trained on the HAM10000 dataset, the model achieves a test accuracy of **78.2%**, contributing to the automation and enhancement of skin cancer diagnosis.

## Features
- **Dataset**: HAM10000, a collection of 10,015 dermoscopic images labeled by expert consensus or histopathological diagnosis.
- **Model Architecture**: Optimized CNN with layers for convolution, pooling, and dropout to extract and classify lesion features.
- **Preprocessing**: Techniques including resizing, normalization, data augmentation, and missing data handling.
- **Visualization**: Performance tracking through accuracy/loss curves and confusion matrices.
- **Applications**: Aids dermatologists by providing a non-invasive diagnostic tool to classify lesions.

## Repository Structure
```
Skin-Cancer-AI/
|
├── data/                     # Placeholder for datasets (not included due to size restrictions)
├── notebooks/
│   └── Project.ipynb         # Jupyter notebook for model training and evaluation
├── src/
│   ├── preprocess.py         # Scripts for preprocessing data
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation script
├── results/
│   ├── accuracy_curves.png   # Accuracy and loss curves
│   └── confusion_matrix.png  # Confusion matrix visualization
├── README.md                 # Project overview and setup instructions
└── requirements.txt          # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or later
- Required packages (see `requirements.txt`):
  ```
  tensorflow
  numpy
  pandas
  matplotlib
  scikit-learn
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Skin-Cancer-AI.git
   cd Skin-Cancer-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add the HAM10000 dataset to the `data/` folder.

### Running the Code
1. Preprocess the data:
   ```bash
   python src/preprocess.py
   ```
2. Train the model:
   ```bash
   python src/train.py
   ```
3. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```

### Results
- **Test Accuracy**: 78.2%
- **Validation Accuracy**: 73.8%

![Accuracy Curves](results/accuracy_curves.png)

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

## License
This project is licensed under the MIT License. See `LICENSE` for details.
