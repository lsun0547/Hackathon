import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import sklearn

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

file_path = '../../Downloads/PythonProjectTestFrontend copy/PythonProjectTestFrontend copy/haunting_dataset.csv'

df = pd.read_csv(file_path)
print(df.head())

df['time'] = df['time'].apply(lambda x: int(x.split(':')[0]))
df['location'] = df['location'].map({
    'North Gate' : 0,
    'Old Oak Tree' : 1,
    'Mausoleum' : 2,
    'Grave 42' : 3,
    'Chapel': 4,
    'Back Fence' : 5})
# Map categorical features to numeric values

df['weather'] = df['weather'].map({
    'Clear': 0,
    'Foggy': 1,
    'Rainy': 2,
    'Stormy': 3,
    'Windy': 4
})

df['moon'] = df['moon'].map({
    'New Moon': 0,
    'Young Moon': 1,
    'Waxing Crescent': 2,
    'Waxing Quarter': 3,
    'Waxing Gibbous': 4,
    'Full Moon': 5,
    'Waning Gibbous': 6,
    'Waning Quarter': 7,
    'Waning Crescent': 8,
    'Old Moon': 9
})

df['temperature'] = df['temperature'].map({
    'Very Cold': 0,
    'Cold': 1,
    'Mild': 2,
    'Warm': 3,
    'Hot': 4
})

df['day'] = df['day'].map({
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
})

df['wind'] = df['wind'].map({
    'None': 0,
    'Light Wind': 1,
    'Steady Wind': 2,
    'Strong Wind': 3,
    'Howling Wind': 4
})

X = df.drop('haunted', axis = 1)
Y = df['haunted']



