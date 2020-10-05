#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization


# In[2]:


#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
type(cancer_dataset)


# In[3]:


# keys in dataset
cancer_dataset.keys()


# In[4]:


# featurs of each cells in numeric format
cancer_dataset['data']


# In[5]:


# malignant or benign value
cancer_dataset['target']


# In[6]:


# target value name malignant or benign tumor
cancer_dataset['target_names']


# In[7]:


# description of data
print(cancer_dataset['DESCR'])


# In[8]:


# name of features
print(cancer_dataset['feature_names'])


# In[9]:


# location/path of data file
print(cancer_dataset['filename'])


# In[10]:


# create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))


# In[11]:


# Head of cancer DataFrame
cancer_df.head(6)


# In[12]:


# Tail of cancer DataFrame
cancer_df.tail(6) 


# In[13]:


# Information of cancer Dataframe
cancer_df.info()


# In[14]:


# Numerical distribution of data
cancer_df.describe()


# In[15]:


#Data Visualization

# Pairplot of cancer dataframe
sns.pairplot(cancer_df, hue = 'target')


# In[16]:


# pair plot of sample feature
sns.pairplot(cancer_df, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )


# In[17]:


# Count the target class
sns.countplot(cancer_df['target'])


# In[18]:


# counter plot of feature mean radius
plt.figure(figsize = (20,8))
sns.countplot(cancer_df['mean radius'])


# In[19]:


# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df)


# In[62]:


#To find a correlation between each feature and target we visualize heatmap
#using the correlation matrix.

# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(15,15))
sns.heatmap(cancer_df.corr(), annot = True, cmap ='plasma', linewidths=3)


# In[21]:


# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)


# In[22]:


# visualize correlation barplot
plt.figure(figsize = (16,5))
ax = sns.barplot(cancer_df2.corrwith(cancer_df.target).index, cancer_df2.corrwith(cancer_df.target))
ax.tick_params(labelrotation = 90)


# In[23]:


#Data Preprocessing
# input variable
X = cancer_df.drop(['target'], axis = 1)
X.head(6)


# In[24]:


# output variable
y = cancer_df['target']
y.head(6)


# In[25]:


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)


# In[26]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[27]:


#After cleaning data, we build the models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[51]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'none')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# In[ ]:





# In[36]:


# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[ ]:





# In[38]:


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)


# In[ ]:





# In[55]:


cm = confusion_matrix(y_test, y_pred_lr)
plt.title('Heatmap of Confusion Matrix (Logistic Regression)', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()


# In[56]:


print(classification_report(y_test, y_pred_lr))


# In[ ]:




