#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




# In[23]:


df=pd.read_csv('diamonds.csv')
df.head()


# In[24]:


df.shape


# In[25]:


#checking for null values
df.info()


# In[26]:


#checking descriptive statistics
df.describe()


# In[27]:


df.head(10)


# In[28]:


sns.histplot(df['price'],bins = 20)


# In[29]:


sns.histplot(df['carat'],bins=20)


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# In[31]:


X=df.drop("price",axis=1)
y=df["price"]


# In[43]:


categorical_cols = ['cut', 'color', 'clarity']   
numeric_cols = ['carat','depth','table','x','y','z'] 


# In[44]:


# Preprocessing: OneHotEncode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])


# In[45]:


# Pipeline with preprocessing + regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# In[46]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:


# Fit model
model.fit(X_train, y_train)


# In[48]:


predict = model.predict(X_test)


# In[49]:


# 9. Evaluate
# -------------------------
print("Mean Squared Error:", mean_squared_error(y_test, predict))
print("R² Score:", r2_score(y_test,predict))


# In[55]:


import streamlit as st


# In[56]:


st.title("💎 Diamond Price Prediction App")

st.write("Enter the diamond details below:")

carat = st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01)
cut = st.selectbox("Cut", df['cut'].unique())
color = st.selectbox("Color", df['color'].unique())
clarity = st.selectbox("Clarity", df['clarity'].unique())
depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, step=0.1)
table = st.number_input("Table %", min_value=40.0, max_value=80.0, step=0.1)
x = st.number_input("Length (x mm)", min_value=2.0, max_value=10.0, step=0.01)
y = st.number_input("Width (y mm)", min_value=2.0, max_value=10.0, step=0.01)
z = st.number_input("Depth (z mm)", min_value=1.0, max_value=10.0, step=0.01)

if st.button("Predict Price"):
    new_diamond = pd.DataFrame([{
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }])

    pred_price = model.predict(new_diamond)[0]
    st.success(f"💰 Predicted Diamond Price: ${pred_price:,.2f}")


# In[ ]:




