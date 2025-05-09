import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
# You can download the dataset from: https://www.kaggle.com/datasets/ludobenistant/hr-analytics
# Place it in the same directory as this script
df = pd.read_csv('HR_comma_sep.csv')

# Convert salary to numeric
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])

# Prepare features and target
X = df.drop('left', axis=1)
y = df['left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Save the model
joblib.dump(model, 'model.joblib')
print("Model saved as 'model.joblib'") 