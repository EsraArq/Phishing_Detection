#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[23]:


#Import the Dataset
df= pd.read_csv(r"C:\Users\18434\Downloads\Phishing_Email.csv\PhishingEmail.csv")
df.head()


# In[27]:


# for data cleaning and prepareation 
# Check for missing values Not A Number"NAN"
# sum() mean count of NAN values of each col of df 
df.isna().sum()


# In[28]:


#Drop tha Na values
# since there is no missing value, the operation does not change the dataframe 
df = df.dropna()
print(df.isna().sum())


# In[29]:


# So far, the data is clean in terms of missing values and I re-checked to drop any NAN values 
# dataset shape attribute to view how many cols and rows in my dataset 
df.shape 
# shows (col, row) 


# In[30]:


# Count the occurrences of each E-mail type in my dataset, counting the number of each unique value 
# this is an important step for classification purpose 
email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)


# In[40]:


# Count the occurrences of each email type
email_type_counts = df['Email Type'].value_counts()

# Create the bar chart with dark red for phishing and light green for safe emails
email_type_counts.plot(kind='bar', color=['lightgreen', 'darkred'])

# Labeling the axes and the title
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types')

# Display the chart
plt.show()


# In[41]:


# We will use undersampling technique // to balance between safe and phishing email for testing purpose 
Safe_Email = df[df["Email Type"]== "Safe Email"]
Phishing_Email = df[df["Email Type"]== "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])


# In[42]:


# checking the shape one more time after undersampling process  
Safe_Email.shape,Phishing_Email.shape


# In[43]:


# creating a new dataset with balanced E-mail type!
Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)
Data.head()


# In[44]:


# NOW THE DATA IS READY, 
# split the data into x and y metrix 
X = Data["Email Text"].values
y = Data["Email Type"].values

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[47]:


#building Random Forest Classifier ML  
# Importing Libraries for the model,Tf-idf and Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# define the Classifier with creating pipline 
classifier = Pipeline([("tfidf",TfidfVectorizer() ),("classifier",RandomForestClassifier(n_estimators=10))])# add another hyperparamters as U want
# Trian Our model
classifier.fit(X_train,y_train)


# In[48]:


# Prediction pn my test using training classifier 
y_pred = classifier.predict(x_test)


# In[49]:


# Importing classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[50]:


#accuracy_score
accuracy_score(y_test,y_pred)


# In[51]:


#confusion_matrix
confusion_matrix(y_test,y_pred)


# In[52]:


#classification_report
classification_report(y_test,y_pred)


# In[54]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

# Assuming y_test and y_pred are already defined

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

# Create a confusion matrix and turn it into a DataFrame
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=['Actual Negative', 'Actual Positive'], 
                              columns=['Predicted Negative', 'Predicted Positive'])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix_df)
print()

# Generate a classification report >> 
class_report = classification_report(y_test, y_pred, target_names=['Phishing Email', 'Safe Email'], output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# To print the classification report
print("Classification Report:")
print(class_report_df)


# In[75]:


# Importing SVM
from sklearn.svm import SVC

#Create the Pipeline
SVM = Pipeline([("tfidf", TfidfVectorizer()),("SVM", SVC(C = 100, gamma = "auto"))])


# In[78]:


# traing the SVM model 
SVM.fit(X_train,y_train)


# In[79]:


# y_pred. for SVM model
s_ypred = SVM.predict(x_test)


# In[80]:


# check the SVM model accuracy
accuracy_score(y_test,s_ypred )


# In[88]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Assuming 'Data' is your preprocessed DataFrame with the 'Email Text' and 'Email Type'
X = Data["Email Text"].values
y = Data["Email Type"].values

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a pipeline with TfidfVectorizer and SVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svc', SVC())
])

# Define a grid of parameters to search (for example)
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['auto', 'scale'],
    'svc__kernel': ['linear', 'rbf']
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best estimator
best_svc = grid_search.best_estimator_

# Predictions
y_pred = best_svc.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)


# In[ ]:




