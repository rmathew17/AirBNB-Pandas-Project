import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



#Age and Gender
file_path = '/Users/riamathew/Downloads/age_gender_bkts.csv'
df_ag = pd.read_csv(file_path)

df_ag['age_bucket'] = df_ag['age_bucket'].apply(lambda x: '100-104' if x == '100+' else x)
#Define mean_age feature
df_ag['mean_age'] = df_ag['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)
df_ag = df_ag.drop('age_bucket', axis=1)
print(df_ag.head())

print(df_ag.isnull().values.any())

print(df_ag['country_destination'].value_counts())

print(df_ag['gender'].value_counts())

print(df_ag['year'].value_counts())


#Sessions
df_sessions = pd.read_csv('/Users/riamathew/Downloads/sessions.csv')
print(df_sessions.head(15))
print(df_sessions.shape)

df_sessions['action'] = df_sessions['action'].replace('-unknown-', np.nan)
df_sessions['action_type'] = df_sessions['action_type'].replace('-unknown-', np.nan)
df_sessions['action_detail'] = df_sessions['action_detail'].replace('-unknown-', np.nan)



print(df_sessions['secs_elapsed'].describe())
print(len(df_sessions[df_sessions['secs_elapsed'].isnull()]))

median_secs = df_sessions['secs_elapsed'].median()
df_sessions['secs_elapsed'] = df_sessions['secs_elapsed'].fillna(median_secs)
print(df_sessions['secs_elapsed'].describe())

print(df_sessions[df_sessions['device_type'].isnull()])

df_sessions['device_type'] = df_sessions['device_type'].replace('-unknown-', np.nan)
print(df_sessions['device_type'].value_counts())




#Countries
countries_path = '/Users/riamathew/Downloads/countries.csv'
df_countries = pd.read_csv(countries_path)
print(df_countries)


#Train

df_train_users = pd.read_csv('/Users/riamathew/Downloads/train_users_2.csv')
print(df_train_users.head())
print(df_train_users.shape)

df_train_users['gender'] = df_train_users['gender'].replace('-unknown-', np.nan)
df_train_users['first_browser'] = df_train_users['first_browser'].replace('-unknown-', np.nan)
print(df_train_users[df_train_users['first_device_type'].isnull()])

print(df_train_users[df_train_users['age'] > 120].head())

df_inf = df_train_users[(df_train_users['country_destination'] != 'NDF') & (df_train_users['country_destination'] != 'other') & (df_train_users['gender'] != 'OTHER') & (df_train_users['gender'].notnull())]
df_inf = df_inf[['id', 'gender', 'country_destination']]
print(df_inf.head())

#Analysis for if there is a relationship between country preference and the sex of the customer.
print(df_inf['gender'].value_counts())
print(df_inf['country_destination'].value_counts())

plt.figure(figsize=(20,8))


sns.barplot(x='mean_age', y='population_in_thousands', hue='gender', data=df_ag, errorbar=None)

sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
pop_stats = df_ag.groupby('country_destination')['population_in_thousands'].sum()
sns.barplot(x=pop_stats.index, y=pop_stats)

