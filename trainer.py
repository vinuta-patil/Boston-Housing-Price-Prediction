import numpy as np
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

# Load dataset
data = pd.read_csv("boston.csv", delimiter=',')
dataset = np.asarray(data)

# Split into input (X) and output (Y) variables
X = dataset[:, 0:13]
Y = dataset[:, 13]

print("Dataset Shape:", dataset.shape)
print("X Shape:", X.shape)
print("Y Shape:", Y.shape)

# Define the baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Set random seed for reproducibility
seed = 7
np.random.seed(seed)

# Wrap model with KerasRegressor
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=1)

# Evaluate model using k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold, scoring="neg_mean_squared_error")

print(f"Results: {results.mean():.4f} ({results.std():.4f}) MSE")

print("FITTING THE MODEL NOW")

# Train model
model = baseline_model()
model.fit(X, Y, batch_size=5, epochs=100, verbose=1)

# Save the model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model.weights.h5")

print("MODEL HAS BEEN SAVED SUCCESSFULLY")
