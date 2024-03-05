import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
import optuna
import coremltools as ct

def objective(trial):
    # Define the hyperparameter space
    param = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.1, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(**param)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    
    # Make predictions and calculate accuracy
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    
    return accuracy

# Load the dataset
data = pd.read_csv('categorized_dataset_nostr.csv')
data['FitnessGoal'] = data['FitnessGoal'].astype('category')
data['FitnessGoal'] = data['FitnessGoal'].cat.codes

# Assuming 'data' is your DataFrame
X = data.drop(columns=['Category']).values
y = data['Category'].values

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)  # You can adjust the number of trials

# Best trial
trial = study.best_trial
print(f"Accuracy: {trial.value:.4f}")
print("Best hyperparameters: ", trial.params)

best_params = trial.params
model = xgb.XGBClassifier(**best_params)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)

# Final evaluation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Final Accuracy: {accuracy * 100.0}%")

choice = input("Do you want to save the model? (y/n): ")
if choice.lower() == 'y':
    coreml_model = ct.converters.xgboost.convert(model)
    coreml_model.save('xgb-cat.mlmodel')
    print("Model saved")
