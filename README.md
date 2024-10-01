# SkinFL
### title: A Lightweight Privacy-Preserving FL for Skin Cancer Diagnosis
![image](https://github.com/user-attachments/assets/b5baf671-bf7f-44dc-a730-38cb5798424b)

## Introduction
Malignant melanoma is highly invasive, with early detection significantly improving survival rates, as early-stage detection offers a five-year survival rate of 99%, which drastically drops to 23% once metastasis occurs. Traditional diagnostic methods like visual exams and biopsies are subjective and limited in accuracy, while machine learning (ML) has enhanced skin cancer diagnosis by automating image analysis, improving objectivity, and even surpassing human specialists in some cases. ML tools also support early detection through patient self-monitoring, helping to prevent advanced-stage cancers. Despite these advancements, ML faces challenges in healthcare, particularly around data fragmentation and privacy concerns. Federated Learning (FL) offers a solution by allowing local model training without sharing raw data, thereby preserving privacy. FL has shown promise in improving cancer diagnosis through collaborations across institutions, but challenges remain, especially related to data heterogeneity and security risks. Cryptographic techniques like secure aggregation, differential privacy, and fully homomorphic encryption (FHE) offer potential solutions, though they come with trade-offs in computational efficiency and scalability. While FHE provides robust security, it imposes high computational costs, and partial homomorphic encryption (PHE) lacks the security and precision necessary for protecting sensitive health data. We therefore propose a Lightweight Privacy-Preserving FL for Skin Cancer Diagnosis
### Prerequisites
This project was developed using Python 3.8. Deep learning was performed using the PyTorch framework. Sections involving cryptography and secure federated learning were developed using OpenMinded's PySyft 0.2.9, TenSeal, and SyMPC.

## Datasets availability
1. [HAM10000 Datasets](https://challenge.isic-archive.com/data/#2018)
2. [ISIC2019 Datasets](https://challenge.isic-archive.com/data/#2019) 
