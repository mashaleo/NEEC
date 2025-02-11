import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('Desktop/NEEC Research/NLP Code/cleaned_and_balanced_complaints_small.csv')

# Convert 'Date received' to datetime and create new features
df['Date received'] = pd.to_datetime(df['Date received'])

# Bin 'Business Days' into 0 days and > 0 days
df['business_days_binned'] = pd.cut(df['Business Days'], 
                                    bins=[-1, 0, df['Business Days'].max()], 
                                    labels=['0 days', '> 0 days'])

# Drop unnecessary columns and handle categorical features
df = df.dropna()  # Remove rows with missing values
X = df.drop(columns=['Date received', 'Business Days', 'business_days_binned'])
X = pd.get_dummies(X)  # Convert categorical variables to numerical
y = df['business_days_binned']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
