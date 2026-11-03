import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into Train, Val, and Test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert numpy arrays to float32 for PyTorch
X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
X_val, y_val = X_val.astype(np.float32), y_val.astype(np.float32)
X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the hyperparameters grid
hidden_sizes = [32]
learning_rates = [0.01]
epochs = [20]

# Initialize variables to store the best model and its metrics
best_model = None
best_val_loss = float('inf')
best_hparams = None
best_train_loss = None
best_val_loss_history = None
best_test_loss = None

# Perform Grid Search for hyperparameters
for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        for epoch in epochs:
            # Define the neural network model
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

            # Define the optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Initialize lists to store the loss values for each epoch
            train_loss_list = []
            val_loss_list = []
            test_loss_list = []

            # Train the model for the specified number of epochs
            for _ in range(epoch):
                # Forward pass
                y_pred_train = model(X_train_tensor)
                loss_train = criterion(y_pred_train, y_train_tensor)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # Evaluate the model on the validation set
                with torch.no_grad():
                    y_pred_val = model(X_val_tensor)
                    loss_val = criterion(y_pred_val, y_val_tensor)

                # Evaluate the model on the test set
                with torch.no_grad():
                    y_pred_test = model(X_test_tensor)
                    loss_test = criterion(y_pred_test, y_test_tensor)

                # Append the loss values to the lists
                train_loss_list.append(loss_train.item())
                val_loss_list.append(loss_val.item())
                test_loss_list.append(loss_test.item())

            # Check if the current model has the lowest validation loss
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                best_model = model
                best_hparams = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'epochs': epoch}
                best_train_loss = train_loss_list
                best_val_loss_history = val_loss_list
                best_test_loss = test_loss_list

# Plot the loss values for the best model
plt.plot(best_train_loss, label='Train MSE')
plt.plot(best_val_loss_history, label='Val MSE')
plt.plot(best_test_loss, label='Test MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Print the best hyperparameters and the corresponding test MSE
print('Best Hyperparameters:', best_hparams)
print('Test MSE:', best_val_loss)