# Imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('spambase.csv')

# Split the data into features and target
X = data.drop('spam', axis=1)
y = data['spam']

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Train the model on the training data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, 'model.pkl')

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy: ', accuracy)

# Plot the confusion matrix and results
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()