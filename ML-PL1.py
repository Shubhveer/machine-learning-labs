#!/usr/bin/env python
# coding: utf-8

# # Machine Learining-Lab

# ## Practical-1

# ### Name:Shubh Veerwani

# ### Class:B_B3

# ### Roll No:34

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Step1: Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

#Display first 5 rows
df.head()


# In[3]:


df.tail()


# In[4]:


print(df.shape)


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[17]:


plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
plt.title("Missing Values")
plt.show()


# In[19]:


#Step 3:Handle missing Values
df["Age"].fillna(df["Age"].median(),inplace=True)


# In[21]:


df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)


# In[23]:


df.isnull().sum()


# In[25]:


df.drop("Cabin",axis=1, inplace=True)


# In[27]:


df.isnull().sum()


# In[29]:


df.duplicated().sum()


# In[31]:


sns.boxplot(x=df["Age"])
plt.show()


# In[33]:


sns.boxplot(x=df["Fare"])
plt.show()


# In[35]:


#Remove Outliers using IQR Method
Q1=df["Fare"].quantile(0.25)
Q3=df["Fare"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

df=df[(df["Fare"]>=lower) & (df["Fare"]<=upper)]


# In[37]:


sns.boxplot(x=df["Fare"])
plt.show()


# In[39]:


#Remove Outliers using IQR Method
Q1=df["Age"].quantile(0.25)
Q3=df["Age"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

df=df[(df["Age"]>=lower) & (df["Age"]<=upper)]


# In[41]:


sns.boxplot(x=df["Age"])
plt.show()


# # Data Encoding

# In[44]:


df["Sex"] = df["Sex"].replace("male",0)
df["Sex"] = df["Sex"].replace("female",1)


# In[46]:


df.head()


# In[48]:


df["Embarked"] = df["Embarked"].replace("S",0)
df["Embarked"] = df["Embarked"].replace("Q",1)
df["Embarked"] = df["Embarked"].replace("C",2)


# In[50]:


df.head()


# In[52]:


df['Embarked'].unique()


# In[58]:


#Univariate Analysis
plt.figure(figsize=(6,4))
sns.histplot(df["Age"],bins=20,kde=True)

plt.title("Age Distribution")
plt.show()


# In[60]:


plt.figure(figsize=(6,4))
sns.histplot(df["Fare"],bins=20,kde=True)

plt.title("Fare Distribution")
plt.show()


# In[64]:


sns.countplot(x="Sex",data=df)
plt.title("Gender Count")
plt.show()


# In[66]:


sns.countplot(x="Pclass",data=df)
plt.title("Gender Count")
plt.show()


# In[68]:


sns.countplot(x="Embarked",data=df)
plt.title("Gender Count")
plt.show()


# In[70]:


#Bivirate Analysis
sns.countplot(x="Sex",hue="Survived",data=df)

plt.title("Gender vs Survived")
plt.show()


# In[72]:


sns.countplot(x="Pclass",hue="Survived",data=df)

plt.title("Class vs Survived")
plt.show()


# In[74]:


sns.countplot(x="Embarked",hue="Survived",data=df)

plt.title("ember vs Survived")
plt.show()


# In[76]:


sns.scatterplot(x="Age",y="Fare",data=df)

plt.title("Age vs Fare")
plt.show()


# In[84]:


#Multivirate Analysis
plt.figure(figsize=(10,8))

numeric_df=df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="inferno")

plt.show()

# viridis,plasmo,inferno,magma,cividis


# In[86]:


#Faeature Scaling
X=df.drop("Survived",axis=1)
y=df["Survived"]


# In[88]:


X=X.drop(["PassengerId","Name","Ticket"],axis=1)


# In[94]:


#Standard Scaling
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X[["Age","Fare"]]=scaler.fit_transform(X[["Age","Fare"]])


# In[96]:


X.head(15)


# In[102]:


#Train Test Spilt
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,train_size=0.8)


# In[104]:


print("Training Data:",X_train.shape)

print("Testing Data:",X_test.shape)


# In[106]:


print(X_train.head())


# In[ ]:




