#!/usr/bin/env python
# coding: utf-8

# # Machine-Learning Lab

# ## Name:Shubh Veerwani

# ## Class:B_B3

# ## Roll No:34

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df=pd.read_csv("USA_Housing.csv")
df.head()


# In[3]:


if 'Address' in df.columns:
    df.drop(['Address'], axis=1, inplace=True)


# In[4]:


df.describe()


# In[5]:


pd.set_option('display.float_format','{:2f}'.format)


# In[6]:


df.isnull().sum()


# In[7]:


plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
plt.title("Missing Values")
plt.show()


# In[8]:


sns.boxplot(x=df["Avg. Area Income"])
plt.show()


# In[9]:


Q1=df["Avg. Area Income"].quantile(0.25)
Q3=df["Avg. Area Income"].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=df[(df["Avg. Area Income"]>=lower) & (df["Avg. Area Income"]<=upper)]
sns.boxplot(x=df["Avg. Area Income"])
plt.show()


# In[10]:


sns.boxplot(x=df["Price"])
plt.show()


# In[11]:


Q1=df["Price"].quantile(0.25)
Q3=df["Price"].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=df[(df["Price"]>=lower) & (df["Price"]<=upper)]
sns.boxplot(x=df["Price"])
plt.show()


# In[12]:


plt.figure(figsize=(8,6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='magma',annot=True, fmt=".2f",linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[13]:


sns.histplot(["Area population"],kde=True)
plt.show()


# In[14]:


plt.figure(figsize=(8,5))
plt.hist(df['Avg. Area Income'], bins=20)
plt.title('Histogram of Avg Area Income')
plt.xlabel('Avg Area Income')
plt.ylabel('Freaquency')
plt.show()


# # SLR

# In[15]:


X= df[['Avg. Area Income']]
y = df['Price']


# In[16]:


from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[17]:


from sklearn.linear_model import LinearRegression 
slr_model = LinearRegression()
slr_model.fit(X_train,y_train)


# In[18]:


y_pred = slr_model.predict(X_test)


# In[19]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

print("Simple Linear Regression")
print("-------------------------")
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", root_mean_squared_error(y_test, y_pred))
print("R2 Score: ",r2_score(y_test, y_pred))


# In[20]:


print("Intercept:",slr_model.intercept_)
print("Slope:", slr_model.coef_[0])


# In[21]:


plt.figure(figsize=(8,6))


# In[22]:


plt.figure(figsize=(6,4))
plt.scatter(X_test,y_test, color='pink',label='Actual data')
plt.plot(X_test, y_pred,color='green', linewidth=2,label='Regression Line')
plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


# In[23]:


income = float(input("Enter Avg Area Income"))
new_data = np.array([[income]])
prediction = slr_model.predict(new_data)
print("Predicted House Price =${:,.2f}".format(prediction[0]))


# # Multiple Linear Regression

# In[28]:


X=df.drop(["Price"],axis=1)

y=df["Price"]


# In[29]:


X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[41]:


mlr_model = LinearRegression()

mlr_model.fit(X_train, y_train)

y_pred = mlr_model.predict(X_test)


# In[42]:


mlr=LinearRegression()

mlr.fit(X_train,y_train)


# In[43]:


coef=pd.DataFrame(
    mlr.coef_,
    X.columns,
    columns=["Coefficient"])
coef


# In[44]:


y_pred=mlr.predict(X_test)


# In[45]:


print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score: ",r2_score(y_test, y_pred))


# In[46]:


predictions = mlr_model.predict(X_test)


# In[47]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test,y=predictions,color='yellow',label='Predictions')
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], color='red',linewidth=2,label='Perfect Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


# In[48]:


income = float(input("Enter Average Area Income: "))
house_age = float(input("Enter Average Area House Age: "))
rooms = float(input("Enter Average Area Number Of rooms: "))
bedrooms = float(input("Enter Average number of bedrooms: "))
population = float(input("Enter Area Population: "))
input_array = [[income,house_age,rooms,bedrooms, population
]]
predicted_price = mlr_model.predict(input_array)

print("Predicted House Price: ${:,.2f}".format(predicted_price[0]))


# # Ridge Regression

# In[50]:


from sklearn.model_selection import GridSearchCV


# In[51]:


param_grid={'alpha' : [0.001,0.01,0.1,10,100]}


# In[52]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[54]:


# Alpha values
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}


# In[55]:


grid_ridge = GridSearchCV(
    estimator=Ridge(),
    param_grid=param_grid,
    scoring='r2',
    cv=5
)


# In[56]:


# Train the model
grid_ridge.fit(X_train, y_train)

# Best Alpha and Best Score
print("Best Alpha:", grid_ridge.best_params_)


# # Lasso Regression

# In[58]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# In[57]:


from sklearn.linear_model import Lasso
param_grid={'alpha':[0.001,0.01,0.1,1,10]}


# In[59]:


grid_lasso=GridSearchCV(Lasso(max_iter=5000),param_grid,cv=5,scoring='r2')

grid_lasso.fit(X_train,y_train)


# In[60]:


print(grid_lasso.best_params_)


# In[61]:


lasso_pred=grid_lasso.predict(X_test)

print(r2_score(y_test,lasso_pred))


# In[63]:


predictions=mlr_model.predict(X_test)


# In[64]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test,y=predictions,color='yellow',label='Predictions')

# Add a reference line (perfect prediction line))
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red',linewidth=2,label='Perfect Fit')

#Add Labels and Legend
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()


# In[ ]:




