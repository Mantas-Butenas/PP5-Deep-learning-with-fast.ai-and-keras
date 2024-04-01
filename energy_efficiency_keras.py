import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load the dataset
energy_efficiency = pd.read_csv('https://archive.ics.uci.edu/static/public/242/data.csv')

# Data Preparation
X = energy_efficiency[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y = energy_efficiency[['Y1', 'Y2']]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the Keras model
model = Sequential([
    Dense(256, activation='tanh', input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='leaky_relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Output layer with 2 neurons for Y1 and Y2
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=80, batch_size=64,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[early_stopping], verbose=1)

# Evaluate the model on the validation set
y_pred = model.predict(X_valid_scaled)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred))

print("RMSE on validation set:", rmse_valid)
