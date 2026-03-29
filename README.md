# Student Pass/Fail Prediction System

## Project Overview
This is a Machine Learning project developed for my AI/ML course. The objective is to build a model that can predict if a student will **Pass** or **Fail** based on their academic performance and study habits.

## Dataset Logic
I created a dataset (`student_data.csv`) that contains the following features for different students:
1. **Attendance_Pct**: The attendance percentage of the student.
2. **Internal_Test1_Mark**: Marks scored in the first internal exam.
3. **Internal_Test2_Mark**: Marks scored in the second internal exam.
4. **Study_Hours_Per_Week**: Number of hours the student studies per week.
5. **Final_Result**: This is the target variable.
   - `1` stands for **Pass**
   - `0` stands for **Fail**

## How the Code Works (Logic)
The code in `main.py` follows these steps:

1. **Loading Data**: It uses the `pandas` library to load the student data from the CSV file.
2. **Feature Selection**: We separate the input features (Attendance, Marks, Hours) from the target result (Pass/Fail).
3. **Splitting Data**: I used `train_test_split` to divide the data. 80% of the data is used to train the machine and 20% is reserved to test if the model is accurate.
4. **Model Training**: I used the **Logistic Regression** algorithm. This algorithm is selected because it is best for "Binary Classification" (choosing between two options like Pass or Fail).
5. **Prediction**: The model makes a prediction on the test data and calculates the accuracy score.

## Software Requirements
* Python 3
* Pandas (for data handling)
* Scikit-learn (for the machine learning model)

## How to Run
1. Install the necessary libraries:
   ```bash
   pip install pandas scikit-learn
