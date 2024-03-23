import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rates):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            layers.append(nn.BatchNorm1d(hidden_size))
            current_size = hidden_size 
        
        layers.append(nn.Linear(hidden_sizes[-1], num_classes)) 
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layers(x)
        return out

data = pd.read_csv('categorized_dataset_nostr.csv')
data['FitnessGoal'] = data['FitnessGoal'].astype('category')
data['FitnessGoal'] = data['FitnessGoal'].cat.codes

X = data.drop(columns=['Category']).values
y = data['Category']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = CustomDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1]
num_classes = len(y.unique())

num_epochs = 10

def objective(trial):
    # Suggest hyperparameters
    hidden_sizes = [
        trial.suggest_int('hidden_size1', 50, 500),
        trial.suggest_int('hidden_size2', 50, 500),
        trial.suggest_int('hidden_size3', 50, 500),
        trial.suggest_int('hidden_size4', 50, 500)
    ]
    dropout_rates = [
        trial.suggest_float('dropout_rate1', 0.1, 0.5),
        trial.suggest_float('dropout_rate2', 0.1, 0.5),
        trial.suggest_float('dropout_rate3', 0.1, 0.5),
        trial.suggest_float('dropout_rate4', 0.1, 0.5)
    ]
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    model = MLP(input_size, hidden_sizes, num_classes, dropout_rates)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
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

# Retrain your model using the best hyperparameters found
best_params = trial.params
hidden_sizes = [best_params['hidden_size1'], best_params['hidden_size2'], best_params['hidden_size3'], best_params['hidden_size4']]
dropout_rates = [best_params['dropout_rate1'], best_params['dropout_rate2'], best_params['dropout_rate3'], best_params['dropout_rate4']]
lr = best_params['lr']

model = MLP(input_size, hidden_sizes, num_classes, dropout_rates)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

data = pd.read_csv('categorized_dataset_nostr.csv')
data['FitnessGoal'] = data['FitnessGoal'].astype('category')
data['FitnessGoal'] = data['FitnessGoal'].cat.codes

X = data.drop(columns=['Category']).values
y = data['Category']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = CustomDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(y.unique())

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
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

    # Evaluate on validation set during training to monitor overfitting
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.2f}%')

# Final evaluation on the validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Final Accuracy on validation set: {final_accuracy:.2f}%')
