

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

# class EnsembleClassificationModel:
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y

#         # Define the 10 machine learning models
#     self.models = [
#         Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", SupportVectorClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", AdaBoostClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", QuadraticDiscriminantAnalysis())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier())]),
#         Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
#     ]

#         # Create the ensemble classifier
#         self.ensemble = StackingClassifier(
#             estimators=self.models, final_estimator=LogisticRegression()
#         )

#     def train(self):
#         # Split the data into training and test sets
#         X_train, X_test, Y_train, Y_test = train_test_split(
#             self.X, self.Y, test_size=0.25, random_state=42
#         )

#         # Fit the ensemble classifier to the training data
#         self.ensemble.fit(X_train, Y_train)

#     def predict(self, X):
#         # Make predictions on the test data
#         Y_pred = self.ensemble.predict(X)