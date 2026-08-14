#!/usr/bin/env python
# coding: utf-8

# # Practical-3 Machine Learning

# ## Name:Shubh Veerwani

# ## Class:B_B3

# ## Roll No:34

# # S-Algorithm

# In[1]:


import pandas as pd
data=pd.read_csv("enjoy.csv")
data


# In[2]:


data.head()


# In[3]:


X=data.iloc[:,:-1].values

y=data.iloc[:,-1].values


# In[4]:


import numpy as np

def find_s(X,y):
    hypothesis=None
    
    print("Intial Hypothesis:",hypothesis)
    for i in range(len(X)):
        if y[i]=="Yes":
            if hypothesis is None:
                hypothesis=X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] !=X[i][j]:
                        hypothesis[j]="?"
        print(f"\nAfter training example{i+1}")
        print(hypothesis)
    return hypothesis
final_hypothesis=find_s(X,y)

print("\nFinal Hypothesis")


# # Candidate Elimination Algorithm

# In[8]:


def candidate_elimination(concepts,target):
    specific=concepts[0].copy()
    general=[["?"for i in range(len(specific))]]
    print("Intial S:",specific)
    print("Intial G:",general)
    for i,h in enumerate(concepts):
        if target[i]=="Yes":
            for x in range(len(specific)):
                if h[x]!=specific[x]:
                    specific[x]="?"
        else:
            for x in range(len(specific)):
                if h[x]!=specific[x]:
                    general.append(["?" if j!=x else specific[x] for j in range(len(specific))])
        print("\nExample",i+1)
        print("S=",specific)
        print("G=",general)
    return specific,general
S,G=candidate_elimination(X,y)

print("\nFinal Specific Boundary")
print(S)
print("\nFinal General Boundary:")
for g in G:
    print(g)


# In[ ]:




