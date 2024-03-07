#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


###Importing the data


# In[3]:


dataset = pd.read_csv(r"J:\1.ML_PROJECT\1.ML_REGGRESION_PROJECTS\3.POLINOMIAL\Position_Salaries.csv")


# In[ ]:


### Read the first 5 rows & colums in the data 


# In[4]:


dataset.head()


# In[ ]:


### data reprocssing 


# In[5]:


dataset.isna().sum()


# In[ ]:


###featuer matrics


# In[6]:


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[13]:


X


# In[15]:


y


# In[ ]:


#ploting the data set for visualization


# In[16]:


sns.pairplot(dataset)


# In[ ]:


# Training the Polynomial Regression model on the whole dataset


# In[35]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[19]:


X_poly


# In[ ]:


# Training the Linear Regression model on the whole dataset


# In[20]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# In[ ]:


# Visualising the Polynomial Regression results


# In[36]:


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# Visualising the Linear Regression results


# In[37]:


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# suppose we have to predict the salary  of a person whose level is 5.5


# In[57]:


p_salary = poly_reg.fit_transform([[5.5]])


# In[58]:


dataset.head(11)


# In[59]:


p_salary


# In[ ]:


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)


# In[60]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# Predicting a new result with Linear Regression


# In[61]:


lin_reg.predict([[6.5]])


# In[ ]:


# Predicting a new result with Polynomial Regression


# In[62]:


lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:




