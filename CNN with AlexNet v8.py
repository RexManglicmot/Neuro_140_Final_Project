#4/8/24
#start -- 6:18 pm
#end -- 9:15 pm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the pre-trained AlexNet model
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # Adjust for binary classification
model.to(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test dataset
test_dataset = datasets.ImageFolder(root='/Users/Rex/PycharmProjects/Neuro 140/test viet', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Path to mixed datasets and results directory
base_path = '/Users/Rex/PycharmProjects/Neuro 140/Mixed DataSets v2'
results_dir = '/Users/Rex/PycharmProjects/Neuro 140/Results v2'
os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists

# Define the training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, roc_auc, conf_mat, all_labels, all_probs

# Iterate over increments and train/evaluate the model
increments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Include 0% increment
results = []

for inc in increments:
    train_dataset_path = os.path.join(base_path, f'train_mixed_{inc}')
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Adjust the number of epochs for the 0% increment
    epochs_to_run = 0 if inc == 0 else 5

    accuracy, precision, recall, f1, roc_auc, conf_mat, labels, probs = train_and_evaluate(model, train_loader, test_loader, device, epochs=epochs_to_run)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues')
    plt.title(f'Confusion Matrix - {inc}% Increment')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{inc}.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_val:.2f}) - {inc}% Increment')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {inc}% Increment')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f'roc_curve_{inc}.png'))
    plt.close()

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    average_precision = average_precision_score(labels, probs)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (AP = {average_precision:.2f}) - {inc}% Increment')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {inc}% Increment')
    plt.legend(loc="best")
    plt.savefig(os.path.join(results_dir, f'precision_recall_curve_{inc}.png'))
    plt.close()

    results.append({
        'Increment': inc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

# Save results to a DataFrame and then to CSV
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(results_dir, 'model_performance v2.csv')
results_df.to_csv(results_csv_path, index=False)
print(results_df)

# Plot accuracy over increments
plt.figure(figsize=(10, 6))
plt.plot(results_df['Increment'], results_df['Accuracy'], marker='o')
plt.title('Accuracy vs Increment Level')
plt.xlabel('Increment Level (%)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'accuracy_over_increments v2.png'))
plt.close()
