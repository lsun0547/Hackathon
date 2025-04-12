import random
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ignore warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier



# Load dataset
file_path = 'haunting_dataset.csv'
df = pd.read_csv(file_path)

#Dropping wind since we found that it did not help much with predictions
#df = df.drop(['wind'], axis = 1)

print(df.head())

# Encode categorical features
df['time'] = df['time'].apply(lambda x: int(x.split(':')[0]))

df['location'] = df['location'].map({
    'North Gate': 0, 'Old Oak Tree': 1, 'Mausoleum': 2,
    'Grave 42': 3, 'Chapel': 4, 'Back Fence': 5})

df['weather'] = df['weather'].map({
    'Clear': 0, 'Foggy': 1, 'Rainy': 2, 'Stormy': 3})

df['moon'] = df['moon'].map({
    'New Moon': 0, 'Young Moon': 1, 'Waxing Crescent': 2, 'Waxing Quarter': 3,
    'Waxing Gibbous': 4, 'Full Moon': 5, 'Waning Gibbous': 6, 'Waning Quarter': 7,
    'Waning Crescent': 8, 'Old Moon': 9})

df['temperature'] = df['temperature'].map({
    'Very Cold': 0, 'Cold': 1, 'Mild': 2, 'Warm': 3, 'Hot': 4})

df['day'] = df['day'].map({
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6})


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

# New LRM

# Logistic Regression

#Creates model type
lr_model = LogisticRegression(random_state=42)
#Puts our TRAINING DATA in the model
lr_model.fit(x_train, y_train)
#Predicts using testing data?
y_pred_lr = lr_model.predict(x_test)

# Logistic regression evaluation
#Checks if above prediction was accurate^^^
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Look at the results of our logistic regression
# Accuracy: Correct predictions / All predictions; how many were right
# Precision: All correct positive predictions / corect positive predictions + incorrect positive predictions; what we though we correct over what was correct
# Recall: Correct positive / Correct positives + incorrect negatives; what we thought were correct over what we shouldve gotten
# F1 Score: (2 x precision x recall) / (precision + recall)
# Confusion Matrix: Top left is true negatives, top right is false positives, bottom left is false negatives, bottom right is true positives; These values are what caluculate the above measures
print(f'Logistic Regression:\n Accuracy: {accuracy_lr}\n Precision: {precision_lr}\n Recall: {recall_lr}\n F1 Score: {f1_lr}')
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# We ran this again with training data instead of testing data. Scores were a bit lower, and thus caused questions around training data, the threshold,
# use of logistic regression, etc.

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

    print(f"Sample {i + 1} - True: {true_str}, Predicted: {pred_str}")

import joblib

# Save the trained model to a file
model_filename = 'ghost_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"\nModel saved to '{model_filename}'")


