#!/usr/bin/env python
# coding: utf-8

# In[1]:


import env
import pandas as pd
import acquire

from sklearn.model_selection import train_test_split


# In[2]:


df = acquire.get_telco_data()


# # Prep Telco
# 
# - drop all 'id' columns
# - created dummy columns for:
#     - contract
#     - internet
#     - payment
# - dropped brand new customers who had no total payment values as they were too new
# - converted total charges to 'float' type
# - encoded binary catagory for:
#     - gender
#     - partner
#     - dependents
#     - phone services
#     - multiple_lines
#     - online_security
#     - online_backup
#     - device_protection
#     - tech_support
#     - streaming_tv
#     - streaming_movies
#     - paperless builing
#     - churn
#     

# In[3]:


def prep_telco(df):
    df.drop(columns = ['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace = True)
    dummy_df = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first= False)
    df = pd.concat([df, dummy_df], axis=1)
    df = df[df.total_charges != ' ']
    df.total_charges = df.total_charges.astype(float)
    
    
    # encode binary categorical variables into numeric values
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service': 3})
    df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    df.drop(columns = 'churn', inplace = True)
    return df


# In[4]:


df = acquire.get_telco_data()


# In[5]:


df = prep_telco(df)


# # Split Telco
# 
# - 20% of data into test group
# - 30% of remaining data into validate group (30% of 80% = 24% of total data)
# - 70% of remaining data into train group (70% of 80% = 56% of total data)
# 

# In[6]:


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn_encoded)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn_encoded)
    return train, validate, test


# In[7]:


train, validate, test = split_telco_data(df)


# In[8]:


df.shape, train.shape, validate.shape, test.shape


# In[ ]:





# In[1]:


def prep_pred_telco(df):
    df.drop(columns = ['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace = True)
    dummy_df = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first= False)
    df = pd.concat([df, dummy_df], axis=1)
    df = df[df.total_charges != ' ']
    df.total_charges = df.total_charges.astype(float)
    
    
    # encode binary categorical variables into numeric values
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service': 3})
    df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 3})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    df.drop(columns = 'churn', inplace = True)
    return df


# In[ ]:




