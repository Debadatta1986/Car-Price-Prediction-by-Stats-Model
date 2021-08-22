#!/usr/bin/env python
# coding: utf-8

# In[872]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cars=pd.read_csv('D:\Debadatta\desktop\data science\CarPrice_Assignment.csv')


# In[873]:


cars.head()


# In[874]:


cars.info()


# In[875]:


#maxda:mazda,Nissan:nissan,porcshce:porsche,toyouta:toyota,  vw,vokswagen:volkswagen,need to change


# In[876]:


cars['car_company']=cars['CarName'].apply(lambda x: x.split(' ')[0])


# In[696]:


cars.head()


# In[698]:


cars['car_company'].loc[(cars['car_company']=='vw') | (cars['car_company']=='vokswagen')]='volkswagen'


# In[700]:


cars['car_company'].loc[cars['car_company']=='Nissan']='nissan'


# In[701]:


cars['car_company']=cars['car_company'].str.replace('maxda','mazda')


# In[702]:


cars['car_company']=cars['car_company'].str.replace('porcshce','porsche').str.replace('toyouta','toyota')


# In[703]:


cars


# In[704]:


cars['car_company'].value_counts()


# In[705]:


cars.drop('CarName',axis=1,inplace=True)


# In[706]:


cars.info()


# In[707]:


cars.cylindernumber.value_counts()


# In[708]:


cars.doornumber.value_counts()


# In[709]:


def num(x):
    return x.map({'four':4,'six':6,'five':5,'eight':8,'two':2,'twelve':12,'three':3})


# In[710]:


cars[['doornumber','cylindernumber']]=cars[['doornumber','cylindernumber']].apply(num)


# In[711]:


cars.info()


# In[712]:


cars_catagorical=cars.select_dtypes(include='object')


# In[713]:


cars_catagorical.head()


# In[714]:


cars_dummies=pd.get_dummies(cars_catagorical,drop_first=True)


# In[715]:


cars_dummies


# In[716]:


cars_catagorical_drop=list(cars_catagorical.columns)


# In[717]:


cars=cars.drop(cars_catagorical_drop,axis=1)


# In[718]:


cars


# In[719]:


cars=pd.concat([cars,cars_dummies],axis=1)


# In[720]:


cars.head()


# In[721]:


cars.drop('car_ID',axis=1,inplace=True)


# In[722]:


cars


# In[723]:


#model buildding


# In[724]:


from sklearn.model_selection import train_test_split


# In[725]:


df_train,df_test=train_test_split(cars,train_size=0.7,test_size=0.3,random_state=75)


# In[726]:


from sklearn.preprocessing import StandardScaler


# In[727]:


scaler=StandardScaler()


# In[728]:


cars.info()


# In[729]:


cars.columns


# In[730]:


varlist=['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth',
       'carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
       'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'highwaympg']


# In[731]:


df_train[varlist]=scaler.fit_transform(df_train[varlist])


# In[732]:


df_train.head()


# In[733]:


y_train=df_train.pop('price')


# In[734]:


y_train


# In[735]:


x_train=df_train


# In[736]:


x_train.info()


# In[737]:


from sklearn.linear_model import LinearRegression


# In[738]:


lm=LinearRegression()


# In[739]:


lm.fit(x_train,y_train)


# In[740]:


lm.coef_


# In[741]:


lm.intercept_


# In[742]:


from sklearn.feature_selection import RFE


# In[743]:


rfe1=RFE(lm,15)
rfe1.fit(x_train,y_train)


# In[744]:


rfe1.ranking_


# In[745]:


rfe1.support_


# In[746]:


import statsmodels.api as sm


# In[747]:


col1=x_train.columns[rfe1.support_]


# In[748]:


col1


# In[749]:


x_train_rfe1=x_train[col1]


# In[750]:


x_train_rfe1


# In[751]:


x_train_rfe1=sm.add_constant(x_train_rfe1)


# In[752]:


x_train_rfe1


# In[753]:


lm1=sm.OLS(y_train,x_train_rfe1).fit()


# In[754]:


lm1.summary()


# In[755]:


vif=pd.DataFrame()


# In[756]:


vif['features']=x_train_rfe1.columns


# In[757]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[758]:


vif["VIF"]=[variance_inflation_factor(x_train_rfe1.values,i) for i in range(x_train_rfe1.shape[1])]


# In[759]:


vif['VIF']=round(vif['VIF'],2)


# In[760]:


vif=vif.sort_values(by='VIF',ascending=False)


# In[761]:


vif


# In[802]:


rfe2=RFE(lm,10)
rfe2.fit(x_train,y_train)


# In[803]:


x_train_rfe2=x_train.columns[rfe2.support_]


# In[804]:


col2=x_train_rfe2


# In[805]:


x_train_rfe2=x_train[col2]


# In[806]:


x_train_rfe2


# In[807]:


x_train_rfe2=sm.add_constant(x_train_rfe2)


# In[808]:


x_train_rfe2


# In[809]:


lm2=sm.OLS(y_train,x_train_rfe2).fit()


# In[810]:


lm2.summary()


# In[811]:


vif=pd.DataFrame()


# In[812]:


vif['features']=x_train_rfe2.columns


# In[813]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[814]:


vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i) for i in range(x_train_rfe2.shape[1])]


# In[815]:


vif['VIF']=round(vif['VIF'],2)


# In[816]:


vif=vif.sort_values(by='VIF',ascending=False)


# In[817]:


vif


# In[818]:


x_train_rfe2.drop('enginelocation_rear',axis=1,inplace=True)


# In[819]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
lm2.summary()


# In[820]:


vif=pd.DataFrame()
vif['features']=x_train_rfe2.columns
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i) for i in range(x_train_rfe2.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif


# In[821]:


x_train_rfe2


# In[822]:


x_train_rfe2.columns


# In[687]:


# features affecting the price of the car 
#'enginesize', 'enginetype_rotor', 'car_company_audi','car_company_bmw', 'car_company_buick', 
#'car_company_jaguar','car_company_porsche', 'car_company_saab', 'car_company_volvo'


# In[ ]:


#making prediction


# In[824]:


df_test[varlist]=scaler.transform(df_test[varlist])


# In[826]:


y_test=df_test.pop('price')


# In[827]:


x_test=df_test


# In[828]:


col2


# In[845]:


x_test_rfe2=x_test[col2]


# In[846]:


x_test_rfe2=x_test_rfe2.drop('enginelocation_rear',axis=1)


# In[847]:


x_test_rfe2=sm.add_constant(x_test_rfe2)


# In[850]:


y_pred=lm2.predict(x_test_rfe2)


# In[851]:


plt.scatter(y_test,y_pred)


# In[853]:


from sklearn.metrics import r2_score


# In[854]:


r2_score(y_test,y_pred)


# In[857]:


col3=col2.drop('enginelocation_rear')


# In[858]:


col3


# In[869]:


plt.figure(figsize=(10,8))
sns.heatmap(cars[col3].corr(),annot=True)


# In[ ]:




