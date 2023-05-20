#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression


# In[6]:


df = pd.read_csv(r"C:\\Users\\raoul\\Downloads\\Simulation population FR Barcelone - Copie de Feuille 4.csv", sep=',')
df


# In[7]:


# Convertir le taux de variation en format numérique
df['taux de variation'] = df['taux de variation'].str.replace(',', '.').str.rstrip('%').astype(float)


# In[8]:


# Diviser les données en deux ensembles : un ensemble avec des valeurs connues et un ensemble avec des valeurs manquantes 
df_known = df.dropna()
df_unknown = df[df['Population FR'].isna()]


# In[9]:


# Créer un modèle de régression linéaire et l'entraîner sur les données connues
X_train = df_known['Année'].values.reshape(-1, 1)
y_train = df_known['Population FR'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


# Effectuer des prédictions pour les valeurs manquantes
X_pred = df_unknown['Année'].values.reshape(-1, 1)
y_pred = regressor.predict(X_pred)

df.loc[df['Population FR'].isna(), 'Population FR'] = y_pred.flatten()


# In[11]:


df


# In[12]:


# calculer les taux de variation manquant
df['taux de variation'] = (df['Population FR'] - df['Population FR'].shift(1)) / df['Population FR'].shift(1) * 100

taux_2020 = df.loc[df['Année'] == 2020, 'taux de variation'].values[0]
taux_2021 = df.loc[df['Année'] == 2021, 'taux de variation'].values[0]
taux_2022 = df.loc[df['Année'] == 2022, 'taux de variation'].values[0]


# In[14]:


# Suppression des lignes avec des valeurs manquantes
df = df.dropna()


# In[17]:


# Arrondir la colonne taux de variation à 2 décimales
df['taux de variation']=round(df['taux de variation'],2)


# In[18]:


# Arrondir la colonne Population FR à 0 décimales
df['Population FR']=round(df['Population FR'],0)


# In[19]:


df


# In[20]:


# Export DF from notebook to CSV
df.to_csv("C:\\Users\\raoul\\Downloads\\barcelona_df_ok.csv")

