import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random


# Ignore warnings
warnings.filterwarnings('ignore')


# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample


# Suppress warnings
warnings.filterwarnings('ignore')


# Load dataset
file_path = 'haunting_dataset.csv'
df = pd.read_csv(file_path)


# Encode categorical features
df['time'] = df['time'].apply(lambda x: int(x.split(':')[0]))


df['location'] = df['location'].map({
   'North Gate': 0, 'Old Oak Tree': 1, 'Mausoleum': 2,
   'Grave 42': 3, 'Chapel': 4, 'Back Fence': 5})


df['weather'] = df['weather'].map({
   'Clear': 0, 'Foggy': 1, 'Rainy': 2, 'Stormy': 3, 'Windy': 4})


df['moon'] = df['moon'].map({
   'New Moon': 0, 'Young Moon': 1, 'Waxing Crescent': 2, 'Waxing Quarter': 3,
   'Waxing Gibbous': 4, 'Full Moon': 5, 'Waning Gibbous': 6, 'Waning Quarter': 7,
   'Waning Crescent': 8, 'Old Moon': 9})


df['temperature'] = df['temperature'].map({
   'Very Cold': 0, 'Cold': 1, 'Mild': 2, 'Warm': 3, 'Hot': 4})


df['day'] = df['day'].map({
   'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
   'Friday': 4, 'Saturday': 5, 'Sunday': 6})


df['wind'] = df['wind'].map({
   'None': 0, 'Light Wind': 1, 'Steady Wind': 2,
   'Strong Wind': 3, 'Howling Wind': 4})


# Balance the dataset
df_majority = df[df.haunted == 0]
df_minority = df[df.haunted == 1]


df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=len(df_majority),
                                random_state=42)


df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)


# Split data
X = df_balanced.drop('haunted', axis=1)
y = df_balanced['haunted']


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a Random Forest classifier with tuned parameters
rf_model = RandomForestClassifier(
   n_estimators=200,
   max_depth=10,
   min_samples_split=5,
   class_weight='balanced',
   random_state=42
)


rf_model.fit(x_train, y_train)


# Predict and evaluate
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)




sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()


# Sample predictions
print("\nSample predictions:")
for i in range(10):
   idx = i * random.randint(1, 10)
   test_sample = x_test.iloc[idx]
   true_label = y_test.iloc[idx]
   predicted_label = rf_model.predict([test_sample])[0]


   true_str = 'Haunted' if true_label == 1 else 'Not Haunted'
   pred_str = 'Haunted' if predicted_label == 1 else 'Not Haunted'


   print(f"Sample {i+1} - True: {true_str}, Predicted: {pred_str}")




import joblib


# Save the trained model to a file
model_filename = 'ghost_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"\nModel saved to '{model_filename}'")