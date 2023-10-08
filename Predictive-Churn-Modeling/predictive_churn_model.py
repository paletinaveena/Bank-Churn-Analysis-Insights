# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data_path = 'data/Churn_Modelling.csv'
df = pd.read_csv(data_path)

# Define the features (independent variables) and the target (dependent variable)
features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']

# Extract features and target variable
X = df[features]
y = df['Exited']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier with optimized hyperparameters
rf_classifier = RandomForestClassifier(
    n_estimators=100,  
    criterion='gini', 
    max_depth=None, 
    min_samples_split=2,  
    min_samples_leaf=1, 
    max_features='auto',
    random_state=42,  
    n_jobs=-1, 
)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)

# Visualize feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=features)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Predictive Model')
plt.show()

# Save the trained model to a file for future use
model_filename = 'churn_prediction_model.pkl'
joblib.dump(rf_classifier, model_filename)
