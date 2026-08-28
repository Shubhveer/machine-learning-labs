#!/usr/bin/env python
# coding: utf-8

# # ML-Practical-4

# ### Name:Shubh Veerwani

# ### Class:B_B3

# ### Roll No:34

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# In[7]:


df=pd.read_csv('winequality-red.csv')


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df.columns


# # Univariate Ananlysis

# In[16]:


#Histogram+KDE
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols=df.columns

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col],kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[17]:


# Boxplot-Outlier Ananlysis
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# In[18]:


# Detect outliers for every feature
features=df.drop(columns=["quality"])

#Calculate Q1 and Q3
Q1=features.quantile(0.25)
Q3=features.quantile(0.75)

# Calculate IQR
IQR=Q3-Q1

#Calculate Lower and Upper Limits
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

# Detect Outliers
outlier_mask=(features<lower_bound) | (features>upper_bound)

#Count outliers for each feature
outlier_count=outlier_mask.sum()

#Percentage of Outliers
outlier_percentage=(outlier_count/len(features))*100

#Create Summary table
outlier_summary=pd.DataFrame({
    'Q1':Q1,
    'Q3':Q3,
    'IQR':IQR,
    'lower bound':lower_bound,
    'upper bound':upper_bound,
    'Outlier Count':outlier_count,
    'Outlier %':outlier_percentage
})

outlier_summary.round(3)


# In[19]:


#No of outliers in each feature

print("Outliers in each feature:\n")

for column in features.columns:
    print(f"{column:25s}:{outlier_count[column]}")


# In[20]:


plt.figure(figsize=(12,6))

outlier_count.sort_values(ascending=False).plot(
    kind='bar'
)

plt.title("No of outliers in each feature")
plt.xlabel("Features")
plt.ylabel("No of Outliers")
plt.xticks(rotation=45,ha='right')
plt.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.show()


# In[21]:


#Quality Distribution
plt.figure(figsize=(7,4))
sns.countplot(x='quality',data=df)
plt.title("Distribution of Wine Quality")
plt.show()


# In[22]:


df['quality'].value_counts().sort_index()


# # 2.Bivariate Analysis

# In[23]:


#Alcohol vs Quality
sns.boxplot(x='quality',y='alcohol',data=df)
plt.title('Alcohol vs Wine Quality')
plt.show()


# In[24]:


features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
          'density','pH','sulphates','alcohol']

for col in features:
    plt.figure(figsize=(7,4))
    sns.boxplot(x='quality',y=col,data=df)
    plt.title(f'{col} vs Wine quality')
    plt.show()


# In[25]:


#Scatter Plot
sns.scatterplot(x='alcohol',y='quality',data=df)
plt.title('Alcohol vs Quality')
plt.show()


# In[26]:


sns.scatterplot(x='volatile acidity',y='quality',data=df)
plt.title('Volatile Acidity vs Quality')
plt.show()


# # 3.Multivariate Analysis

# In[27]:


plt.figure(figsize=(12,8))

corr=df.corr()

sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')

plt.title('Correlation Heatmap')
plt.show()


# In[28]:


# Pair Plot
selected=['alcohol','volatile acidity','sulphates','citric acid','quality']

sns.pairplot(df[selected],hue='quality')

plt.show()


# In[29]:


#Multivariate Ananlysis Using Quality Groups
df['quality_group']=pd.cut(
    df['quality'],
    bins=[0,5,6,10],
    labels=['Low','Medium','High']
)


# In[30]:


sns.boxplot(
    x='quality_group',
    y='alcohol',
    data=df
)

plt.title('Alcohol Distribution Across Quality Groups')
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
                              f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)


# In[32]:


X=df.drop('quality',axis=1)
y=df['quality']


# In[33]:


print(y.value_counts())


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                                               random_state=42,
                                               stratify=y
                                              )


# In[35]:


criterion='entropy'


# In[36]:


dt_model=DecisionTreeClassifier(
    criterion='entropy',
    random_state=42)


# In[37]:


df['quality_group']=pd.cut(
    df['quality'],
    bins=[0,5,6,10],
    labels=['Low','Medium','High']
)


# In[38]:


print(X.columns)


# In[39]:


print(X.select_dtypes(exclude='number').columns)


# In[40]:


y=df['quality']


# In[41]:


X=df.drop(columns=['quality','quality_group'],errors='ignore')


# In[42]:


print(X.columns)
print(X.dtypes)


# In[43]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                                               random_state=42,
                                               stratify=y
                                              )


# In[44]:


df['quality_group']=pd.cut(
    df['quality'],
    bins=[0,5,6,10],
    labels=['Low','Medium','High']
)


# In[45]:


print(X.columns)


# In[46]:


dt_model=DecisionTreeClassifier(criterion='entropy',random_state=42)

dt_model.fit(X_train,y_train)


# In[47]:


dt_model.fit(X_train,y_train)


# In[48]:


y_train_pred=dt_model.predict(X_train)
y_test_pred=dt_model.predict(X_test)


# In[49]:


train_accuracy=accuracy_score(y_train,y_train_pred)

print("Training Accuracy:",train_accuracy)


# In[50]:


test_accuracy=accuracy_score(y_test,y_test_pred)

print("Testing Accuracy:",test_accuracy)


# In[51]:


print("Precision:",
      precision_score(y_test,y_test_pred,average='weighted'))
print("Recall:",
      recall_score(y_test,y_test_pred,average='weighted'))
print("F1 Score:",
      f1_score(y_test,y_test_pred,average='weighted'))


# In[52]:


print(classification_report(y_test,y_test_pred))


# In[53]:


#Confusion Matrix
cm=confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(8,6))

sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')

plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('confusion matrix-decision tree')

plt.show()


# # Decision Trees

# In[54]:


print("Tree Depth:",dt_model.get_depth())
print("Number of Leaves:",dt_model.get_n_leaves())


# In[55]:


plt.figure(figsize=(25,15))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Entropy")
plt.show()


# In[56]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(24,12))

plot_tree(dt_model,feature_names=X.columns,
          class_names=[str(c) for c in sorted(y.unique())],filled=True,
          rounded=True,
          impurity=True,precision=2,fontsize=10,max_depth=3
         )

plt.title("Decision Tree- Top 3 levels",fontsize=18)
plt.tight_layout()
plt.show()


# # AUC ROC CURVE

# In[57]:


# ROC Curve and AUC-Befpre hypermater tuning
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

#Get unique classes
classes=sorted(y_test.unique())

# Convert y_test to binary format
y_test_bin=label_binarize(y_test,classes=classes)

#Get probability predictions
y_score=dt_model.predict_proba(X_test)

#Plot ROC Curve
plt.figure(figsize=(10,7))

for i,class_label in enumerate(classes):
    fpr,tpr,thresholds=roc_curve(y_test_bin[:,i],y_score[:,i])

    roc_auc=auc(fpr,tpr)

    plt.plot(fpr,tpr,linewidth=2,label=f'Quality{class_label} (AUC={roc_auc:2f})')

#Random Classifier
plt.plot([0,1],[0,1],linestyle='--',label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve-Before Hyperparameter tuning')

plt.legend()
plt.grid()
plt.show()

#Overall AUC
overall_auc_before=roc_auc_score(y_test_bin,y_score,multi_class='ovr',
                                average='weighted'
                                )

print("Overall ROC-AUC Befor Tuning:",round(overall_auc_before,4))


# # Hyperparameter Tuning

# In[58]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

param_grid={
    'criterion':['entropy'],
    'max_depth':[3,5,7,10,None],
    'min_samples_split':[2,5,10,20],
    'min_samples_leaf':[1,2,4,8],
    'max_features':[None,'sqrt','log2'],
    'ccp_alpha':[0,0.001,0.005,0.01]
}

grid=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,
                  cv=5,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

print("Best Parameters")
print(grid.best_params_)

print("Best Cross Validation Accuracy")
print(grid.best_score_)

best_tree=grid.best_estimator_


# In[59]:


y_pred=best_tree.predict(X_test)

print(classification_report(y_test,y_pred))
print("Accuracy=",accuracy_score(y_test,y_pred))


# In[60]:


#Training accuracy
train_accuracy=best_tree.score(X_train,y_train)

#Testing accuracy
test_accuracy=best_tree.score(X_test,y_test)

print("Training Accuracy=",train_accuracy)
print("Testing Accuracy=",test_accuracy)


# In[61]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(25,12))

plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=[str(c) for c in best_tree.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree After Hyperparameter Tuning")
plt.show()


# # Cross Validation

# In[62]:


from sklearn.model_selection import cross_val_score

scores=cross_val_score(best_tree,X,y,cv=10,scoring="accuracy")

print(scores)

print("Average Accuracy=",scores.mean())

print("Std:",scores.std())


# # Learning Curve

# In[63]:


from sklearn.model_selection import learning_curve

train_sizes,train_scores,test_scores=learning_curve(best_tree,X,y,cv=5,scoring='accuracy')

plt.figure(figsize=(8,5))

plt.plot(train_sizes,train_scores.mean(axis=1),marker='o',label='Training Accuracy')

plt.plot(train_sizes,test_scores.mean(axis=1),marker='o',label='Validation Accuracy')

plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")

plt.legend()

plt.show()


# # For new data sample

# In[68]:


new_wine = pd.DataFrame([{
    'Id':1,
    'fixed acidity':7.7,
    'volatile acidity':0.7,
    'citric acid':0.40,
    'residual sugar':2.0,
    'chlorides':0.06,
    'free sulfur dioxide':36.0,
    'total sulfur dioxide':45.0,
    'density':1.30,
    'pH':3.4,
    'sulphates':0.55,
    'alcohol':9.7
}])

new_wine = new_wine[best_tree.feature_names_in_]

prediction = best_tree.predict(new_wine)

print("Predicted wine quality:", prediction[0])


# In[ ]:




