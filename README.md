# Classification-of-Schizophrenia-for-Early-Diagnosis

# Schizophrenia Classification using 1D CNN and Bi-LSTM Models

## Project Overview
This repository contains implementations of deep learning models for the early diagnosis of schizophrenia using EEG (Electroencephalogram) signals. The project focuses on using advanced neural network architectures to automatically extract significant features from EEG data and perform binary classification between schizophrenic and non-schizophrenic subjects.

## Background
Schizophrenia is a severe mental health disorder characterized by abnormal behaviors and difficulty distinguishing between reality and imagination. Early and accurate diagnosis is crucial for effective treatment, as approximately 2.4 million adults in the United States are affected, with over 21 million cases worldwide. Traditional diagnosis methods rely heavily on subjective expert visualization of behavioral markers, which can be inconsistent and potentially miss sub-clinical brain abnormalities.

This project leverages EEG signals, which capture brain electrical activity, as a non-invasive and cost-effective approach for schizophrenia detection.

## Dataset
- EEG records from 33 healthy adolescents and 45 adolescents with diagnosed schizophrenia
- EEG data sampled at 128 Hz
- Obtained through a conventional 10-20 configuration for electrode placement
- Processed into 858 segments, each with 1280×16 sample points (20s duration with 5s overlap)

## Data Preprocessing
The preprocessing pipeline consists of:

1. **Filtering**
   - High-pass filter (0.5 Hz) to remove low-frequency noise (head movements, eye blinks)
   - Notch filter to eliminate power-line interference (50/60 Hz)

2. **Segmentation**
   - Filtered signals segmented into overlapping windows (20s duration, 5s overlap)
   - Each segment contains 1280×16 sample points

## Implemented Models

### 1. 1D Convolutional Neural Network (1D CNN)
- Architecture with progressively increasing filter complexity (64 → 128 → 256 → 512)
- Kernel size of 3 with stride 1
- Batch normalization, LeakyReLU activation, and pooling layers
- Achieves 99.41% validation accuracy

### 2. ResNet + BiLSTM
- ResNet-based feature extractor with residual blocks
- Bidirectional LSTM for temporal pattern recognition in both directions
- Global average pooling and dense layers with ReLU activation
- Achieves 98.83% validation accuracy

### 3. ResNet + BiLSTM + Attention
- Extends the ResNet+BiLSTM model with attention mechanisms
- Focuses on critical parts of the sequence
- Enhanced feature representation
- Achieves 99.42% validation accuracy

### 4. Transformer Model
- Conv1D preprocessing layer
- Multiple Transformer blocks with multi-head attention
- Captures global temporal dependencies without BiLSTM
- Achieves 98.84% validation accuracy


## Key Findings
- The 1D CNN model offers the best balance of performance and computational efficiency
- ResNet-based models demonstrate the ability to capture deeper temporal features
- Attention mechanisms slightly improve performance by focusing on critical sequence elements
- All models significantly outperform traditional machine learning approaches

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

### Data Preparation
```python
# Example code for data loading and preprocessing
import numpy as np

# Load EEG data
eeg_data = np.load('eeg_dataset.npy')

# Apply filtering
from utils.preprocessing import apply_filters
filtered_data = apply_filters(eeg_data, highpass_cutoff=0.5, notch_freq=50)

# Segment data
from utils.preprocessing import segment_signals
segmented_data = segment_signals(filtered_data, window_size=20, overlap=5, sampling_rate=128)
```

## Future Work
- Ensemble modeling to exploit the benefits of different architectures
- Multimodal fusion combining EEG with fMRI or genetic data
- Explainable AI methods to identify critical EEG features for diagnosis
- Deployment as a clinical decision support tool

## References
1. Oh, S. L., Vicnesh, J., Ciaccio, E. J., Yuvaraj, R., & Acharya, U. R. (2019). Deep convolutional neural network model for automated diagnosis of Schizophrenia using EEG signals. Applied Sciences, 9(14), 2870.
2. Febles, E. S., Ortega, M. O., Sosa, M. V. & Sahli, H. (2022). Machine learning techniques for the diagnosis of schizophrenia based on event related potentials. medRxiv.
3. De Rosa, A. et al. (2022). Machine learning algorithm unveils glutamatergic alterations in the postmortem schizophrenia brain. Schizophrenia, 8(1), 1-16.

## Authors
- Ilakkia Bharathi B - ilakkiabharathi.b@gmail.com
- Varsha V - vijayalayanvarsha@gmail.com
- Venkatesh M - venkat011003@gmail.com
- Vijay Jeyakumar - vijayj@ssn.edu.in

## Acknowledgments
- Department of Biomedical Engineering, Sri Sivasubramaniya Nadar College of Engineering, Chennai, India
