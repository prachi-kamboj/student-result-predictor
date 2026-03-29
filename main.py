import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('student_data.csv')

# Select features and target
X = data[['Attendance_Pct', 'Internal_Test1_Mark', 'Internal_Test2_Mark', 'Study_Hours_Per_Week']]
y = data['Final_Result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Predict for a new student
print("Prediction for new student:")
new_data = [[79, 17, 20, 8]]
result = model.predict(new_data)
print(result)
