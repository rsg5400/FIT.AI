import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna

# Load and preprocess the data
data = pd.read_csv('categorized_dataset_nostr.csv')

# Encode the 'FitnessGoal' column
label_encoder = LabelEncoder()
data['FitnessGoal'] = label_encoder.fit_transform(data['FitnessGoal'])

# Define features and target
X = data[['Protein', 'Calories', 'Fat', 'Carbohydrates', 'FitnessGoal']].values
y = data['Category'].values

# Initialization parameters
input_size = 5  # Number of features
num_classes = len(data['Category'].unique())
num_epochs = 20

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader for training and validation sets
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, x):
        # Use self.hidden_size here
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

criterion = nn.CrossEntropyLoss()

def objective(trial):
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = RNNModel(input_size, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Adjust DataLoader batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"Best trial accuracy: {trial.value:.4f}")
print("Best hyperparameters: ", trial.params)

# Retrain with best hyperparameters
best_hidden_size = trial.params['hidden_size']
best_lr = trial.params['lr']
best_batch_size = trial.params['batch_size']

# Reinstantiate dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

# Train the model
model = RNNModel(input_size=5, hidden_size=best_hidden_size, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=best_lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Validation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
