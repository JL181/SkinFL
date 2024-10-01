# SkinFL
### title: A Lightweight Privacy-Preserving FL for Skin Cancer Diagnosis
![image](https://github.com/user-attachments/assets/b5baf671-bf7f-44dc-a730-38cb5798424b)

## Introduction
Malignant melanoma is highly invasive, with early detection significantly improving survival rates, as early-stage detection offers a five-year survival rate of 99%, which drastically drops to 23% once metastasis occurs. Traditional diagnostic methods like visual exams and biopsies are subjective and limited in accuracy, while machine learning (ML) has enhanced skin cancer diagnosis by automating image analysis, improving objectivity, and even surpassing human specialists in some cases. ML tools also support early detection through patient self-monitoring, helping to prevent advanced-stage cancers. Despite these advancements, ML faces challenges in healthcare, particularly around data fragmentation and privacy concerns. Federated Learning (FL) offers a solution by allowing local model training without sharing raw data, thereby preserving privacy. FL has shown promise in improving cancer diagnosis through collaborations across institutions, but challenges remain, especially related to data heterogeneity and security risks. Cryptographic techniques like secure aggregation, differential privacy, and fully homomorphic encryption (FHE) offer potential solutions, though they come with trade-offs in computational efficiency and scalability. While FHE provides robust security, it imposes high computational costs, and partial homomorphic encryption (PHE) lacks the security and precision necessary for protecting sensitive health data. We therefore propose a Lightweight Privacy-Preserving FL for Skin Cancer Diagnosis
### Prerequisites
This project was developed using Python 3.8. Deep learning was performed using the PyTorch framework. Sections involving cryptography and secure federated learning were developed using OpenMinded's PySyft 0.2.9, TenSeal, and SyMPC.

## Code Structure Overview

### 1. Cryptographic
- **PFHE.py**: This module contains the implementation of the Packed Fully Homomorphic Encryption (PFHE) scheme, essential for encrypted data aggregation and inference.

### 2. PPFL (Privacy-Preserving Federated Learning)
- **enc_Aggregation.py**: Handles the encrypted aggregation logic using PFHE, ensuring local updates from clients are securely aggregated without exposing raw data.
- **federated_training.py**: Manages the overall federated training process, coordinating training across multiple clients, managing communication, and ensuring the model's privacy.

### 3. network
- **DoConv.py**: Implements the Depthwise Over-parameterized Convolution (DO-Conv) layer, improving model performance, particularly on non-IID data during local training.
- **network.py**: Defines the architecture of the deep neural network (DNN) used in the framework, integrating DO-Conv to enhance accuracy and performance during training and inference.

### 4. prognosis
- **enc_Inference.py**: Handles encrypted inference, ensuring predictions are made while keeping patient data encrypted, safeguarding privacy.
- **enc_network.py**: Contains network configuration and management for encrypted computations, especially for prediction purposes.

### 5. preprocess.py
- Handles data preprocessing and augmentation, preparing the skin cancer datasets for training and evaluation.


## Datasets Availability

We utilized two authoritative multiclass dermoscopic image datasets, HAM10000 and ISIC2019, to evaluate our proposed FL method for the detection of pigmented skin lesions.

1. [HAM10000 Datasets](https://challenge.isic-archive.com/data/#2018)  
   The HAM10000 dataset consists of 10,015 dermoscopic images, categorized into seven types of skin lesions: melanoma, vascular lesions, benign keratosis lesions, dermatofibroma, melanocytic nevi, basal cell carcinoma, and actinic keratoses. All images were verified through histopathology, in vivo confocal microscopy, expert consensus, or follow-up assessments, ensuring high data quality and reliability.

2. [ISIC2019 Datasets](https://challenge.isic-archive.com/data/#2019)  
   The ISIC2019 dataset, being larger and more diverse than HAM10000, comprises 25,331 dermoscopic images and includes an additional class—squamous cell carcinoma—which is absent in HAM10000. This expanded dataset enables a broader evaluation across a diverse range of skin lesions.

