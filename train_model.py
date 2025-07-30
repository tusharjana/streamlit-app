import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. Load and Prepare the Data ---

# Load the dataset
try:
    df = pd.read_csv('student_performance.csv')
except FileNotFoundError:
    print("Error: file not found.")
    print("Please make sure the dataset file is in the same directory as this script.")
    exit()


# Clean up column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop the Student_ID as it's not a predictive feature
df = df.drop('Student_ID', axis=1)

# --- 2. Feature Engineering and Encoding ---

# Convert categorical variables into dummy/indicator variables.
# This is a crucial step for the machine learning model.
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop('Grade_BA', axis=1, errors='ignore') # Drop one of the grade columns created by get_dummies
X = X.drop('Grade_BB', axis=1, errors='ignore')
X = X.drop('Grade_CB', axis=1, errors='ignore')
X = X.drop('Grade_CC', axis=1, errors='ignore')
X = X.drop('Grade_Fail', axis=1, errors='ignore')


# The original 'Grade' column is what we want to predict.
y = df['Grade']

# Align columns after one-hot encoding - crucial for prediction later
# We need to ensure the app sends data with the exact same columns as the training data
X_cols = X.columns

# --- 3. Train the Machine Learning Model ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the RandomForestClassifier
# n_estimators is the number of trees in the forest.
# random_state ensures reproducibility.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate the Model ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")


# --- 5. Save the Model and Columns ---

# We save the trained model and the list of columns to a file using pickle.
# The Streamlit app will load this file to make predictions.
model_data = {
    "model": model,
    "columns": X_cols
}

with open('student_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model and column data saved to 'student_model.pkl'")
print("\nYou can now run the Streamlit app using: streamlit run app.py")

