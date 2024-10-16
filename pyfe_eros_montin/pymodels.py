

import pandas as pd
#from sklearn.ensemble import VotingClassifier
        # Create a VotingClassifier meta-estimator
#        self.voting_classifier = VotingClassifier(estimators=self.classifiers, voting='hard')


import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class EnsembleClassificationModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        # Define the 10 different machine learners
        self.base_estimators = [
            LogisticRegression(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            SVC(),
            MLPClassifier(),
            LogisticRegression(penalty='l1'),
            KNeighborsClassifier(n_neighbors=10),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(n_estimators=100),
            SVC(kernel='linear'),
            MLPClassifier(hidden_layer_sizes=[100, 50]),
        ]

    def train(self):
        # Create a StackingClassifier object
        self.stacking_classifier = StackingClassifier(
            estimators=self.base_estimators,
            final_estimator=MLPClassifier(hidden_layer_sizes=[100, 50]),
        )

        # Train the StackingClassifier object
        self.stacking_classifier.fit(self.X, self.Y)

    def predict(self, X_test):
        # Make predictions using the trained StackingClassifier object
        predictions = self.stacking_classifier.predict(X_test)

        return predictions
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class EnsembleClassificationModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        
        self.models = [
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]),
        Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", SupportVectorClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", AdaBoostClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        Pipeline([("scaler", StandardScaler()), ("clf", QuadraticDiscriminantAnalysis())]),
        Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier())]),
        Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
    ]

        # Create the ensemble classifier
        self.ensemble = StackingClassifier(
            estimators=self.models, final_estimator=LogisticRegression()
        )

    def train(self):
        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=0.25, random_state=42
        )

        # Fit the ensemble classifier to the training data
        self.ensemble.fit(X_train, Y_train)

    def predict(self, X):
        # Make predictions on the test data
        Y_pred = self.ensemble.predict(X)
        return Y_pred
    
    def evaluate(self, X_test, Y_test):
        # Evaluate the ensemble classifier on the test data
        accuracy = self.ensemble.score(X_test, Y_test)
        
        return accuracy
    
    def get_models(self):
        return self.models
    def get_ensemble(self):
        return self.ensemble
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class CNNWithScalars(nn.Module):
    def __init__(self, img_shape, n_scalars):
        super(CNNWithScalars, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate the flattened size after convolution and pooling
        self.flatten_size = 128 * (img_shape[1] // 8) * (img_shape[2] // 8)  # Output size after 3 pooling layers
        
        # Fully connected layers (for combined CNN and scalar features)
        self.fc1 = nn.Linear(self.flatten_size + n_scalars, 64)  # Combine image and scalar features
        self.fc2 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x_img, x_scalars):
        # CNN branch
        x = self.pool(torch.relu(self.conv1(x_img)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(-1, self.flatten_size)  # Flatten
        
        # Concatenate scalar inputs with CNN output
        x_combined = torch.cat((x, x_scalars), dim=1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x_combined))
        x = torch.sigmoid(self.fc2(x))
        return x

# Step 2: Simulate the data (medical images and scalar features)
n_samples = 100
img_height, img_width = 128, 128
n_channels = 1  # Grayscale images
n_scalars = 5  # Number of scalar features (e.g., age, clinical data, etc.)

# Simulated medical images and scalar features
X_images = np.random.rand(n_samples, n_channels, img_height, img_width).astype(np.float32)
X_scalars = np.random.rand(n_samples, n_scalars).astype(np.float32)

# Labels (binary classification)
y = np.random.randint(0, 2, size=n_samples).astype(np.float32)

# Train-test split
X_train_img, X_test_img, X_train_scalars, X_test_scalars, y_train, y_test = train_test_split(
    X_images, X_scalars, y, test_size=0.2, random_state=42)

# Step 3: Define the training function
def train(model, criterion, optimizer, X_train_img, X_train_scalars, y_train, epochs=5):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Convert data to tensors
        inputs_img = torch.tensor(X_train_img)
        inputs_scalars = torch.tensor(X_train_scalars)
        labels = torch.tensor(y_train).view(-1, 1)

        # Forward pass
        outputs = model(inputs_img, inputs_scalars)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 4: Define the evaluation function
def evaluate(model, X_test_img, X_test_scalars, y_test):
    model.eval()
    with torch.no_grad():
        inputs_img = torch.tensor(X_test_img)
        inputs_scalars = torch.tensor(X_test_scalars)
        labels = torch.tensor(y_test).view(-1, 1)
        
        outputs = model(inputs_img, inputs_scalars)
        predicted = (outputs > 0.5).float()  # Apply threshold for binary classification
        
        accuracy = accuracy_score(labels, predicted)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Step 5: Train and Evaluate the Model
input_shape = (n_channels, img_height, img_width)
model = CNNWithScalars(img_shape=input_shape, n_scalars=n_scalars)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, criterion, optimizer, X_train_img, X_train_scalars, y_train, epochs=10)

# Evaluate the model
evaluate(model, X_test_img, X_test_scalars, y_test)

        