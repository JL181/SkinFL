import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import random

# Load
file_path_hmnist = './data/hmnist_28_28.csv'
data_hmnist = pd.read_csv(file_path_hmnist)
original_dataset_size = len(data_hmnist)
print(f'Original dataset size: {original_dataset_size}')

images = data_hmnist.iloc[:, :-1].values
labels = data_hmnist['label'].values

def augment_image(image):
    img = image.reshape(28, 28).astype(np.uint8)

    # Random rotation between -5 and 5 degrees
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((14, 14), angle, 1)
    img = cv2.warpAffine(img, M, (28, 28))

    # Random horizontal flip
    if random.random() < 0.1:
        img = cv2.flip(img, 1)

    # Random vertical flip
    if random.random() < 0.1:
        img = cv2.flip(img, 0)

    return img.flatten()

augmented_images = []
augmented_labels = []

for i in range(len(images)):
    for _ in range(10):
        augmented_image = augment_image(images[i])
        augmented_images.append(augmented_image)
        augmented_labels.append(labels[i])

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)
augmented_dataset_size = len(augmented_labels)
augmented_data = np.column_stack((augmented_images, augmented_labels))
augmented_df = pd.DataFrame(augmented_data, columns=[f'pixel{str(i).zfill(4)}' for i in range(784)] + ['label'])
augmented_df.to_csv('./data/hmnist_28_28_aug.csv', index=False)


torch.manual_seed(73)

# Load the dataset
file_path = './data/hmnist_28_28_aug.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train.reshape((-1, 1, 28, 28)), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.reshape((-1, 1, 28, 28)), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

selected_indices = torch.randperm(len(train_dataset))[:80000]
selected_train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

train_loader = DataLoader(selected_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def split_dataset(dataset, proportions):
    assert sum(proportions) == 1, "Proportions must sum to 1."
    lengths = [int(len(dataset) * p) for p in proportions]
    # 确保所有长度的和等于数据集的总长度（可能有浮点数误差）
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    subsets = torch.utils.data.random_split(dataset, lengths)
    return subsets

train_subsets = split_dataset(train_dataset, [0.1, 0.35, 0.25, 0.15, 0.15])

for i, subset in enumerate(train_subsets):
    print(f"Subset {i + 1} size: {len(subset)}")

local_loaders = [DataLoader(subset, batch_size=64, shuffle=True) for subset in train_subsets]
