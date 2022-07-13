#!/usr/bin/env python
# coding: utf-8

# In[1]:


import env
import pandas as pd


# In[2]:


#url pulls data from env.py file


# In[3]:


url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/telco_churn'


# In[4]:


#query to pull telco data from mysql


# In[5]:


query = '''
SELECT 
    *
FROM
    customers
        JOIN
    contract_types USING (contract_type_id)
        JOIN
    internet_service_types USING (internet_service_type_id)
        JOIN
    payment_types USING (payment_type_id);
'''


# In[6]:


telco = pd.read_sql(query, url)


# In[7]:


telco.head()


# In[8]:


telco.shape


# In[9]:


#turn it into a function


# In[10]:


def new_telco_data():
    return pd.read_sql('''SELECT 
            *
            FROM
            customers
                JOIN
            contract_types USING (contract_type_id)
                JOIN
            internet_service_types USING (internet_service_type_id)
                JOIN
            payment_types USING (payment_type_id)''', url)


import os

def get_telco_data():
    filename = "telco.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_telco = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_telco.to_csv(filename)

        # Return the dataframe to the calling code
        return df_telco


# In[11]:


#verify it works


# In[12]:


df = new_telco_data()


# In[13]:


df.shape


# In[14]:


df.head()


# In[15]:


df = get_telco_data()


# In[16]:


df.shape


# In[17]:


df.head()


# In[ ]:




