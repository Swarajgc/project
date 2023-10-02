#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import missingno
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

warnings.simplefilter("ignore")
sns.set_style("whitegrid")


# In[37]:


df = pd.read_csv('IMDb Movies India.csv',encoding='latin-1')


# In[38]:


df.head()


# In[39]:


df.shape


# In[40]:


missingno.matrix(df)


# In[41]:


year = []
for y in df.Year:
    if type(y) == float:
        year.append(np.nan)
    else:
        year.append(int(str(y)[1:5]))
df["Year"] = year
df.head()


# In[42]:


duration = []
for d in df.Duration:
    if type(d) == float:
        duration.append(np.nan)
    else:
        duration.append(int(str(d).split(" ")[0]))
df["Duration"] = duration
df.head()


# In[43]:


df["Votes"]=df["Votes"].replace("$5.16M", 516)
df["Votes"] = pd.to_numeric(df['Votes'].str.replace(',',''))
df.head()


# In[44]:


df.info()


# In[46]:


df.describe().T


# In[47]:


df.isnull().sum()


# In[48]:


def TopTenPlot(column):
    global df
    df[column].value_counts().sort_values(ascending=False)[:10].plot(kind="bar", figsize=(20,6), edgecolor="k")
    plt.xticks(rotation=0)
    plt.title("Top Ten {}".format(column))
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()


# In[49]:


def Histogram(column):
    global df
    plt.figure(figsize=(20,6))
    plt.hist(df[column], edgecolor="k")
    plt.xticks(rotation=0)
    plt.title("Histogram of {}".format(column))
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# In[50]:


def Scatter(x, y, c=None):
    global df
    plt.figure(figsize=(20,6))
    plt.scatter(df[x], df[y], edgecolor="k", c=c)
    plt.xticks(rotation=0)
    plt.title("Scatter plot X:{} / Y:{}".format(x, y))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show() 


# In[51]:


TopTenPlot("Director")


# In[52]:


TopTenPlot("Actor 1")


# In[53]:


TopTenPlot("Actor 2")


# In[54]:


TopTenPlot("Actor 3")


# In[55]:


Histogram("Year")


# In[56]:


Histogram("Duration")


# In[57]:


Histogram("Votes")


# In[58]:


Histogram("Rating")


# In[59]:


from itertools import combinations
 
comb = combinations(["Duration", "Year", "Rating", "Votes"], 2)
comb_list = []
for i in list(comb):
    comb_list.append(list(i))
    
for col in comb_list:
    Scatter(col[0], col[1])


# In[60]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(method='spearman'),annot=True,cmap="Blues", fmt='.0%')
plt.show()


# In[61]:


worst = df.sort_values("Votes", ascending=False).dropna().tail(10)[[ "Name", "Year", "Duration", "Votes", "Director","Actor 1"]].reset_index(drop=True)
top = df.sort_values("Votes", ascending=False).head(10)[[ "Name", "Year", "Duration", "Votes", "Director","Actor 1"]].reset_index(drop=True)


# In[63]:


worst


# In[64]:


top


# In[65]:


plt.figure(figsize=(20,6),dpi=100)
plt.scatter(top["Year"], top["Duration"], edgecolor="k",s=150, label="Top 10")
for i in range(10):
    plt.text(x=top["Year"][i]+0.8,y=top["Duration"][i]-1.2,s=top["Name"][i], fontsize=11, rotation =0)
plt.scatter(worst["Year"], worst["Duration"], color="red",s=150, label="Worst 10", edgecolor="k")
for i in range(10):
    plt.text(x=worst["Year"][i]-1.8,y=worst["Duration"][i]+3,s=worst["Name"][i], fontsize=11, rotation =0)
plt.legend()
plt.title("The 10 movies with the most votes (with Movie Name)", fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.xticks(rotation=0)
plt.show()


# In[66]:


plt.figure(figsize=(20,6),dpi=100)
plt.scatter(top["Year"], top["Duration"], edgecolor="k",s=150, label="Top 10")
for i in range(10):
    plt.text(x=top["Year"][i]+0.8,y=top["Duration"][i]-1.2,s=top["Director"][i], fontsize=11, rotation =0)
plt.scatter(worst["Year"], worst["Duration"], color="red",s=150, label="Worst 10", edgecolor="k")
for i in range(10):
    plt.text(x=worst["Year"][i]-1.8,y=worst["Duration"][i]+3,s=worst["Director"][i], fontsize=11, rotation =0)
plt.legend()
plt.title("The 10 movies with the most votes (with Director Name)", fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.xticks(rotation=0)
plt.show()


# In[67]:


plt.figure(figsize=(20,6),dpi=100)
plt.scatter(top["Year"], top["Duration"], edgecolor="k",s=150, label="Top 10")
for i in range(10):
    plt.text(x=top["Year"][i]+0.8,y=top["Duration"][i]-1.2,s=top["Actor 1"][i], fontsize=11, rotation =0)
plt.scatter(worst["Year"], worst["Duration"], color="red",s=150, label="Worst 10", edgecolor="k")
for i in range(10):
    plt.text(x=worst["Year"][i]-1.8,y=worst["Duration"][i]+3,s=worst["Actor 1"][i], fontsize=11, rotation =0)
plt.legend()
plt.title("The 10 movies with the most votes (with Lead Role Name)", fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.xticks(rotation=0)
plt.show()


# In[ ]:




