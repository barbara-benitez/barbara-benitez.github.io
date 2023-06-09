#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Project
# this project will look at a dataset with various characteristics of wine to determine if the wine is of good quality

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ### Read in the csv file

# In[2]:


df = pd.read_csv("winequality-red.csv")


# ## EDA 
# ### Look at the data; the target variable is quality

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


np.shape(df)


# In[7]:


df.describe()


# ## Check for missing data

# In[8]:


df.isnull().sum()


# In[9]:


# look at some line plots with the quality rating
# write a loop to generate the 11 plots
cols = (len(df.columns))-1
for col in range(0,cols) :
    
    plt.figure(figsize = (20,6))
    sns.lineplot(x='quality', y=df.columns[col], data = df)
    plt.title(f"{df.columns[col]} versus quality", fontsize = 16)
    plt.show()


# In[10]:


# look for correlations in the data
correlations = df.corr(method='pearson')
plt.figure(figsize=(16,12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[11]:


# Iterate through the correlation matrix and print out columns having correlation above .6
print("The following column pairs have a correlation above 0.6:\n")
for col1 in correlations.columns:
    for col2 in correlations.columns:
        if col1 != col2 and correlations.loc[col1, col2] > 0.6:
            print(f"     {col1} and {col2}: {correlations.loc[col1, col2]}\n")


# - During the pre-processing phase we will drop the columns for fixed acidity, and free sulfur dioxide due to the high correlation with other features
# 

# In[12]:


# Create a clean copy of the data
df_clean = df.copy()


# ## Pre-process the data
# - Drop the columns for fixed acidity and free sulfur dioxide
# - This will entail rescaling the data using StandardScaler

# In[13]:


# drop
df.drop(["fixed acidity", "free sulfur dioxide"], axis=1, inplace=True)


# In[14]:


df.head()


# ### Since quality is a numeric value, and we are trying to classify the wine as above average or not above average we need to designate a new column for above or below average. We will designate high_quality =1 for wines with a quality at or above 6 or a 0 if it is below the threshold. (Note that the designates high_quality to be in the 50th percentile or higher)

# In[33]:


# Create high_quality column based on the condition that it is a 1 if quality 
#scores is greater than 6 using a lambda function
df['high_quality'] = df['quality'].apply(lambda x: 0 if x < 6 else 1)


# In[34]:


df.head()


# In[36]:


## Drop the quality column as it is now encoded in high_quality
df.drop(["quality"],axis = 1, inplace=True)


# ### Set up the feature and target arrays

# In[ ]:





# In[38]:


X = df.iloc[:, :-1]
X.head()


# In[39]:


y = df.iloc[:,-1]
y.head()


# ## Set up the training and test data

# In[40]:


X_test, X_train, y_test, y_train = train_test_split(X, y, random_state =4, test_size =.2)


# In[ ]:





# In[ ]:





# ### Rescale the data using StandardScaler

# In[41]:


scaler = StandardScaler().fit(X_train)


# In[42]:


X_train_scaled = scaler.transform(X_train)


# ## Run the Naive Bayes algorithm 

# In[43]:


model = GaussianNB()


# In[44]:


# Fit the Naive Bayes model to the training data
model.fit(X_train, y_train)


# In[24]:


y_pred = model.predict(X_test)


# In[47]:


# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Model Accuracy:", accuracy)


# ## Decision Tree Classification

# In[48]:


from sklearn.tree import DecisionTreeClassifier


# In[49]:


# create instance of Decision Tree Classifier
dt = DecisionTreeClassifier()


# In[50]:


# fit the training data
dt.fit(X_train, y_train)


# In[51]:


# run the predictions
y_pred_dt = dt.predict(X_test)


# In[54]:


accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)


# ## KNN Classification

# In[55]:


from sklearn.neighbors import KNeighborsClassifier


# In[56]:


knn = KNeighborsClassifier()


# In[57]:


knn.fit(X_train, y_train)


# In[59]:


y_pred_knn = knn.predict(X_test)


# In[60]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score", accuracy_knn)


# ## Logistic Regression Classification

# In[61]:


from sklearn.linear_model import LogisticRegression


# In[64]:


lg = LogisticRegression(max_iter=1000)


# In[65]:


lg.fit(X_train, y_train)


# In[66]:


y_pred_lg = lg.predict(X_test)


# In[67]:


accuracy_lg = accuracy_score(y_test, y_pred_lg)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)


# ## Ridge Classification

# In[68]:


from sklearn.linear_model import RidgeClassifier


# In[69]:


ridge = RidgeClassifier()


# In[70]:


ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)


# In[72]:


accuracy_ridge = accuracy_score(y_test, y_pred_ridge)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)


# ## Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[76]:


rf = RandomForestClassifier()


# In[77]:


rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[78]:


accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)


# ## XGBoost Classifier

# In[81]:


from xgboost import XGBClassifier


# In[82]:


xgb = XGBClassifier()


# In[83]:


xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


# In[84]:


accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)


# ## Basic Neural Network

# In[86]:


from sklearn.neural_network import MLPClassifier


# In[89]:


mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter = 1000, activation='relu', solver='adam', random_state = 12)


# In[90]:


mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)


# In[91]:


accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)
print("Neural Network Classifier Accuracy Score: ", accuracy_mlp)


# ## SVM Classifier

# In[93]:


from sklearn.svm import SVC


# In[94]:


svc = SVC ()


# In[95]:


svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)


# In[97]:


accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("Naive Bayes Model Accuracy:", accuracy)
print("Decision Tree Classifier Accuracy Score: ", accuracy_dt)
print("KNN Classifier Accuracy Score: ", accuracy_knn)
print("Logistic Regression Accuracy Score: ", accuracy_lg)
print("Ridge Classifier Accuracy Score: ", accuracy_ridge)
print("Random Forest Accuracy Score: ", accuracy_rf)
print("XGBoost Classifier Accuracy Score: ", accuracy_xgb)
print("Neural Network Classifier Accuracy Score: ", accuracy_mlp)
print("SVM Classifier Accuracy Score: ", accuracy_svc)


# In[ ]:




