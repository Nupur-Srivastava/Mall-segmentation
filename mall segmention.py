#!/usr/bin/env python
# coding: utf-8

# In[87]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


df = pd.read_csv("Mall_Customers.csv")
df.head()


# In[89]:


inputs = df.drop('CustomerID',axis='columns')
target = df.Gender
inputs.head()


# In[91]:


inputs = inputs.drop('Gender',axis='columns')
inputs.head()


# In[93]:


dummies = pd.get_dummies(target)
dummies.head(4)


# In[96]:


outputs=pd.concat([inputs,dummies],axis='columns')
outputs.head(4)


# In[97]:


plt.scatter(outputs.Age,outputs['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')


# In[98]:


plt.scatter(outputs.Age,outputs['Annual Income (k$)'])
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')


# # K means for age x spending score

# In[99]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(outputs[['Age','Spending Score (1-100)']])
y_predicted


# In[103]:


outputs['cluster']=y_predicted
outputs.head()


# In[106]:


outputs1 = df[outputs.cluster==0]
outputs2 = df[outputs.cluster==1]
outputs3 = df[outputs.cluster==2]
plt.scatter(outputs1.Age,outputs1['Spending Score (1-100)'],color='green')
plt.scatter(outputs2.Age,outputs2['Spending Score (1-100)'],color='red')
plt.scatter(outputs3.Age,outputs3['Spending Score (1-100)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()


# In[109]:


scaler = MinMaxScaler()

scaler.fit(outputs[['Spending Score (1-100)']])
df['Spending Score (1-100)'] = scaler.transform(outputs[['Spending Score (1-100)']])

scaler.fit(outputs[['Age']])
df['Age'] = scaler.transform(outputs[['Age']])


# In[110]:


df.head()


# In[112]:


plt.scatter(df.Age,df['Spending Score (1-100)'])


# In[114]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Spending Score (1-100)']])
y_predicted


# In[115]:


df['cluster']=y_predicted
df.head()


# In[116]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Spending Score (1-100)'],color='green')
plt.scatter(df2.Age,df2['Spending Score (1-100)'],color='red')
plt.scatter(df3.Age,df3['Spending Score (1-100)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[119]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Spending Score (1-100)']])
    sse.append(km.inertia_)


# In[139]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# # K means for age x annual income

# In[124]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(outputs[['Age','Annual Income (k$)']])
y_predicted


# In[126]:


outputs['cluster']=y_predicted
outputs.head()


# In[127]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Annual Income (k$)'],color='green')
plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[129]:


scaler = MinMaxScaler()

scaler.fit(outputs[['Annual Income (k$)']])
df['Annual Income (k$)'] = scaler.transform(outputs[['Annual Income (k$)']])

scaler.fit(outputs[['Age']])
df['Age'] = scaler.transform(outputs[['Age']])


# In[130]:


df.head()


# In[131]:


plt.scatter(df.Age,df['Annual Income (k$)'])


# In[132]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Annual Income (k$)']])
y_predicted


# In[133]:


df['cluster']=y_predicted
df.head()


# In[134]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Annual Income (k$)'],color='green')
plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[137]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Annual Income (k$)']])
    sse.append(km.inertia_)


# In[138]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:




