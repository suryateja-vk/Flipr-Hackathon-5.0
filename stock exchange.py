#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
import numpy as np
import seaborn as sns
from keras.layers import Dropout
import re
from fancyimpute import KNN  
from sklearn.preprocessing import StandardScaler


# In[2]:


data_train = pd.read_excel("/Users/suryateja/Downloads/Train_dataset.xlsx")
data_put_call=pd.read_excel("/Users/suryateja/Downloads/Test_dataset.xlsx",sheet_name="Put-Call_TS")
data_test=pd.read_excel("/Users/suryateja/Downloads/Test_dataset.xlsx")


# In[5]:


def index(col):  
    if col=="NYSE":
        return 1
    if col == "BSE":
        return 2
    if col == "S&P 500":
        return 3
    if col == "NSE":
        return 4
    if col=="JSE":
        return 5
def genindex(col):
    a = col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1:
            return 12765.84
        if a ==2:
            return 38182.08
        if a==3:
            return 3351.28
        if a==4:
            return 11270.15
        if a==5:
            return 55722 
    else:
        return b
def doler(col):
    a = col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1 or a==3:
            return 1
        if a ==2 or a==4:
            return 74.9
        if a ==5:
            return 17.7
    else:
        return b
def industry(col):
    if col=="Real Estate":
        return 1
    if col=="Information Tech":
        return 2
    if col=="Materials":
        return 3
    if col=="Healthcare":
        return 4
    if col=="Energy":
        return 5
def covid(col):
    a =col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1:
            return -0.43
        if a==2:
            return 0.23
        if a==3:
            return 0.03
        if a==4:
            return 0.78
        if a==5:
            return 0.11
    else:
        return b

data_train['Covid Impact (Beta)'] = data_train[['Industry','Covid Impact (Beta)']].apply(covid,axis=1)

data_train['Industry'] = data_train['Industry'].apply(industry)

data_train['Dollar Exchange Rate'] = data_train[['Index','Dollar Exchange Rate']].apply(doler,axis=1)

data_train['General Index'] = data_train[['Index','General Index']].apply(genindex,axis = 1)

data_train['Index'] = data_train['Index'].apply(index)

data_test['Covid Impact (Beta)'] = data_test[['Industry','Covid Impact (Beta)']].apply(covid,axis=1)

data_test['Industry'] = data_test['Industry'].apply(industry)

data_test['Dollar Exchange Rate'] = data_test[['Index','Dollar Exchange Rate']].apply(doler,axis=1)

data_test['General Index'] = data_test[['Index','General Index']].apply(genindex,axis = 1)

data_test['Index'] = data_test['Index'].apply(index)


# In[6]:


data_train = data_train.drop(columns=['Stock Index'])
data_test = data_test.drop(columns=['Stock Index'])
data_put_call = data_put_call.drop(columns=['Stock Index'])


# In[14]:


new_header = data_put_call.iloc[0] 
data_put_call= data_put_call[1:] 
data_put_call.columns = [1,2,3,4,5,6]


# In[7]:


data_train.replace(r'^\s*$', np.nan, regex=True);
data_train[:] = KNN(k=3).fit_transform(data_train) ;


# In[8]:


data_test.replace(r'^\s*$', np.nan, regex=True);
data_test[:] = KNN(k=3).fit_transform(data_test) ;


# In[23]:


data_put_call.head()


# In[15]:


data_put_call.replace(r'^\s*$', np.nan, regex=True);
data_put_call[:] = KNN(k=3).fit_transform(data_put_call) ;


# In[31]:


data_test.info()


# In[32]:


y = data_train['Stock Price']
X = data_train.drop(columns = ['Stock Price'])
X_TEST=data_test

sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_TEST=sc_X.fit_transform(X_TEST)


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=101)


# In[62]:


model = Sequential()
model.add(Dense(64, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(32,kernel_initializer='normal', activation="relu"))
model.add(Dense(16,kernel_initializer='normal', activation="relu"))
model.add(Dense(1, kernel_initializer='normal', activation="linear"))
model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer='adam',metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0,validation_data=(X_test, y_test))
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)


# In[63]:


y_predct_test=model.predict(X_TEST,verbose=0)


# In[128]:


X_put = data_put_call.drop(columns = [6])
y_put =data_put_call[6]
A=data_put_call.drop(columns=[1])
sc_X = StandardScaler()
X_put_train = sc_X.fit_transform(X_put_train)
A = sc_X.fit_transform(A)


# In[129]:


X_put_train,X_put_test,y_put_train,y_put_test = train_test_split(X_put,y_put,test_size=0.30,random_state=101)


# In[130]:


model2 = Sequential()
model2.add(Dense(64, input_dim=5, kernel_initializer='normal', activation='relu'))
model2.add(Dense(32,kernel_initializer='normal', activation="relu"))
model2.add(Dense(8,kernel_initializer='normal', activation="relu"))
model2.add(Dense(1, kernel_initializer='normal', activation="linear"))
model2.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer='adam',metrics=['accuracy'])
model2.fit(X_put, y_put, epochs=1000, verbose=0,validation_data=(X_put_test, y_put_test))
train_mse = model2.evaluate(X_put_train, y_put_train, verbose=0)
test_mse = model2.evaluate(X_put_test, y_put_test, verbose=0)


# In[131]:


B=model2.predict(A,verbose=0)


# In[132]:


data_test["Put-Call Ratio"]=B
final_pre=sc_X.fit_transform(data_test)


# In[133]:


final_predict=model.predict(final_pre)


# In[158]:


solution2=pd.read_excel("/Users/suryateja/Downloads/Test_dataset.xlsx")


# In[159]:


solution2=pd.DataFrame(solution2.iloc[:,0])


# In[160]:


file=pd.DataFrame(final_predict,columns=[['Stock Price(16th August)']])


# In[161]:


solution2[['Stock Price(16th August)']]=file


# In[163]:


solution2.to_csv("solution2data.csv")

