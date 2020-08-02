#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#TASK 1: IMPORT LIBRARIES

import numpy as np
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)


# In[3]:


#TASK 2: LOAD THE DATA - The adverstiting dataset captures sales revenue generated with respect to advertisement spends across 
                         #multiple channles like radio, tv and newspaper.

advert=pd.read_csv("Advertising.csv")
advert.head()


# In[4]:


advert.info()


# In[5]:


#TASK 3: RELATIONSHIP BETWEEN FEATURES AND RESPONSE

sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7)


# In[6]:


#TASK 4: MULTIPLE LINEAR REGRESSION - ESTIMATING COEFFICIENTS

from sklearn.linear_model import LinearRegression

x = advert[['TV', 'radio', 'newspaper']]
y = advert.sales

lm1=LinearRegression()
lm1.fit(x,y)

print(lm1.intercept_)
print(lm1.coef_)


# In[8]:


list(zip(['TV','radio', 'newspaper'], lm1.coef_))


# In[10]:


sns.heatmap(advert.corr(), annot=True)


# In[14]:


#TASK 5: FEATURE SELECTION - Here, model fit and the accuracy of the predictions will be evaluated using R² also known as r2_score

from sklearn.metrics import r2_score

lm2=LinearRegression().fit(x[['TV','radio']], y)
lm2_pred=lm2.predict(x[['TV','radio']])

print("R^2 = ", r2_score(y,lm2_pred))


# In[15]:


lm3=LinearRegression().fit(x[['TV','radio','newspaper']], y)
lm3_pred=lm3.predict(x[['TV','radio','newspaper']])

print("R^2 = ", r2_score(y,lm3_pred))


# In[26]:


#Task 6: Model Evaluation Using Train/Test Split and Metrics

#Mean Absolute Error (MAE) is the mean of the absolute value of the errors:
#Mean Squared Error (MSE) is the mean of the squared errors:
#Root Mean Squared Error (RMSE) is the mean of the squared errors:

#Here, let's use train/test split with RMSE to see whether newspaper should be kept in the model:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x=advert[['TV','radio','newspaper']]
y=advert.sales

x_train, x_test, y_train, y_test= train_test_split(x,y, random_state=1)

lm4 = LinearRegression().fit(x_train,y_train)
lm4_pred=lm4.predict(x_test)

print("RMSE = ", np.sqrt(mean_squared_error(y_test,lm4_pred)))
print("R^2 = ", r2_score(y_test,lm4_pred))


# In[27]:


x=advert[['TV','radio']]
y=advert.sales

x_train, x_test, y_train, y_test= train_test_split(x,y, random_state=1)

lm5 = LinearRegression().fit(x_train,y_train)
lm5_pred=lm5.predict(x_test)

print("RMSE = ", np.sqrt(mean_squared_error(y_test,lm5_pred)))
print("R^2 = ", r2_score(y_test,lm5_pred))


# In[30]:


from yellowbrick.regressor import PredictionError, ResidualsPlot

visualizer=PredictionError(lm5).fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.show()


# In[32]:


#TASK 7: INTERACTION EFFECT - SYNERGY

advert['interaction']= advert['TV'] * advert['radio']

x=advert[['TV', 'radio', 'interaction']]
y=advert.sales

x_train, x_test, y_train, y_test= train_test_split(x,y, random_state=1)

lm6 = LinearRegression().fit(x_train,y_train)
lm6_pred=lm6.predict(x_test)

print("RMSE = ", np.sqrt(mean_squared_error(y_test,lm6_pred)))
print("R^2 = ", r2_score(y_test,lm6_pred))


# In[34]:


visualizer=PredictionError(lm6).fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.show()


# In[ ]:


# CONCLUSION: The goal of this prject was to identify trivial features. 
#Lesser the RMSE value, higher is the accuracy. 
#Thus, in this model, newspaper seems to be a less important feature.

