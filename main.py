import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
data = pd.read_csv('student_data.csv')

# 2. Select features and target
feature_cols = ['Attendance_Pct', 'Internal_Test1_Mark', 'Internal_Test2_Mark', 'Study_Hours_Per_Week']
X = data[feature_cols]
y = data['Final_Result']

# 3. Split the data (Added random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model (Added max_iter for better convergence)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6. Test the accuracy
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# 7. Predict for a new student
print("\nPrediction for new student:")

# Package new data as a DataFrame to avoid feature name warnings
new_student = pd.DataFrame([[79, 17, 20, 8]], columns=feature_cols)

# Remember to scale the new data using the same scaler!
new_student_scaled = scaler.transform(new_student)

result = model.predict(new_student_scaled)
print(f"Predicted Result: {result[0]}")
