import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna

# Define the desired column types
dtype_dict = {
    'restaurant_name': str, 
    'menu_item': str, 
    'Protein': int, 
    'Calories': int, 
    'Fat': int, 
    'Carbohydrates': int, 
    'FitnessGoal': str, 
    'Category': int}

# Read CSV into dataframe
data = pd.read_csv('categorized_dataset_nostr.csv', dtype=dtype_dict)

# Encode variables
data['FitnessGoal'] = data['FitnessGoal'].astype('category')
data['FitnessGoal'] = data['FitnessGoal'].cat.codes

# Split Dataset into inputs and outputs
X = data.drop(columns=['Category'])
Y = data['Category']

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define custom PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader for training and validation sets
train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Modify the Net class to accept dynamic parameters
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        z = torch.relu(self.input(x))
        z = torch.relu(self.hidden2(z))
        z = torch.relu(self.hidden3(z))
        z = self.out(z)
        return z

def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int('hidden_size', 50, 500)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True) 
    
    # Model, criterion, and optimizer
    model = Net(input_size=5, hidden_size=hidden_size, out_size=len(np.unique(Y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


# Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
