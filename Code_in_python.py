# Import required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate a larger sample dataset for medical diagnosis
np.random.seed(42)

# Generating features (X) with 100 instances and 5 features
X = np.random.rand(100000, 5)

# Generating labels (y) randomly as 0 or 1
y = np.random.randint(2, size=100000)

# Define the BLB sampling function
def blb_sampling(X_train, y_train, sample_size):
    num_instances = X_train.shape[0]
    indices = np.random.choice(num_instances, size=sample_size, replace=True)
    X_blb = X_train[indices]
    y_blb = y_train[indices]
    return X_blb, y_blb

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ensemble of BLB-RF models
num_models = 10
sample_size = int(0.8 * X_train.shape[0])

models = []
for i in range(num_models):
    # Perform BLB sampling on the training data
    X_train_blb, y_train_blb = blb_sampling(X_train, y_train, sample_size)

    # Train a Random Forest model on the BLB sample
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_blb, y_train_blb)

    # Add the trained model to the ensemble
    models.append(model)

# Make predictions on the testing data using the ensemble
ensemble_predictions = []
for model in models:
    predictions = model.predict(X_test)
    ensemble_predictions.append(predictions)

# Combine the predictions from the ensemble using voting or averaging
final_predictions = ensemble_predictions[0]
for i in range(1, num_models):
    final_predictions += ensemble_predictions[i]
final_predictions = final_predictions.astype(float)
final_predictions /= num_models

# Evaluate the accuracy of the ensemble predictions
accuracy = accuracy_score(y_test, final_predictions.round().astype(int))
print("Accuracy of BLB-RF:", accuracy)

# Train a standard Random Forest model on the full training set
standard_model = RandomForestClassifier(n_estimators=100)
standard_model.fit(X_train, y_train)

# Make predictions on the testing data using the standard RF model
standard_predictions = standard_model.predict(X_test)

# Evaluate the accuracy of the standard RF model
standard_accuracy = accuracy_score(y_test, standard_predictions)
print("Accuracy of Standard RF:", standard_accuracy)

