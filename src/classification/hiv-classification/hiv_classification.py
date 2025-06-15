import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_hiv(hiv_path):
    ''' Cleaning hiv.csv after 7 molecules failed '''
    df = pd.read_csv(hiv_path)

    # Identify failed molecules
    failed_indices = []
    failed_smiles = []

    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_indices.append(idx)
            failed_smiles.append(smiles)

    print(f"Found {len(failed_indices)} failed molecules:")
    for i, smile in enumerate(failed_smiles, 1):
        print(f"{i}. {smile}")

    # Create a cleaned dataframe without the failed molecules
    cleaned_df = df.drop(failed_indices)

    # Save the cleaned dataframe
    cleaned_df.to_csv('hiv_cleaned.csv', index=False)
    print("\nSaved cleaned dataset to 'hiv_cleaned.csv'")
    print(f"Original count: {len(df)}, Cleaned count: {len(cleaned_df)}")

    cleaned_df = pd.read_csv('hiv_cleaned.csv')
    return cleaned_df

def prepare_data(df, embeddings):
    ''' Helper function to prepare data for all classifiers '''
    X = embeddings
    y = df['HIV_active'].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def random_forest(df, embeddings):
    ''' Random Forest Classifier with detailed metrics '''
    X_train, X_test, y_train, y_test = prepare_data(df, embeddings)

    print("Class distribution:", pd.Series(y_train).value_counts())

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\nDetailed Random Forest Classifier Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def neural_network(df, embeddings):
    ''' Neural Network Classifier '''
    X_train, X_test, y_train, y_test = prepare_data(df, embeddings)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Define the neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)  # 2 output classes
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Initialize model
    model = SimpleNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nTraining Neural Network:")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs, 1)
            acc = (predicted == y_test_t).float().mean()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)
        y_pred = predicted.numpy()
        y_prob = F.softmax(test_outputs, dim=1)[:, 1].numpy()

    print("\nNeural Network Classifier Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    df = clean_hiv('hiv.csv')
    embeddings = torch.load('gnn_embedings.pt').numpy()
    
    # Run all classifiers
    random_forest(df, embeddings)
    neural_network(df, embeddings)