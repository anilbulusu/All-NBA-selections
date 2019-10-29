#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA


# In[3]:


# Importing the dataset
data = pd.read_csv('NBA.csv')


# In[4]:


data.head()


# In[5]:


del data['#']


# In[6]:


data.head()


# In[7]:


data = data.fillna(0)


# In[8]:


data.describe()


# In[9]:


test = pd.DataFrame()
test = data


# In[10]:


test.dtypes


# In[11]:


cols = []


# In[12]:


cols = test.columns
print(len(cols))


# In[13]:


cols = cols[1:22]


# In[14]:


cols


# In[15]:


for i in cols:
    test[i]=test[i].astype('float')


# In[16]:


test.dtypes


# In[17]:


scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(test[['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','EFG%','TS%']])
scaled_df = pd.DataFrame(scaled_df, columns=['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','EFG%','TS%'])


# In[18]:


#Before Scaling Vs After Scaling

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(30, 10))

#Before Scaling
ax1.set_title('Before Scaling')
sns.kdeplot(test['GP'], ax=ax1)
sns.kdeplot(test['MIN'], ax=ax1)
sns.kdeplot(test['PTS'], ax=ax1)
sns.kdeplot(test['FGM'], ax=ax1)
sns.kdeplot(test['FGA'], ax=ax1)
sns.kdeplot(test['FG%'], ax=ax1)
sns.kdeplot(test['3PM'], ax=ax1)
sns.kdeplot(test['3PA'], ax=ax1)
sns.kdeplot(test['3P%'], ax=ax1)
sns.kdeplot(test['FTM'], ax=ax1)
sns.kdeplot(test['FTA'], ax=ax1)
sns.kdeplot(test['FT%'], ax=ax1)
sns.kdeplot(test['OREB'], ax=ax1)
sns.kdeplot(test['DREB'], ax=ax1)
sns.kdeplot(test['REB'], ax=ax1)
sns.kdeplot(test['AST'], ax=ax1)
sns.kdeplot(test['STL'], ax=ax1)
sns.kdeplot(test['BLK'], ax=ax1)
sns.kdeplot(test['TOV'], ax=ax1)
sns.kdeplot(test['EFG%'], ax=ax1)
sns.kdeplot(test['TS%'], ax=ax1)

#After Scaling
ax2.set_title('After Min-Max Scaling')
sns.kdeplot(scaled_df['GP'], ax=ax2)
sns.kdeplot(scaled_df['MIN'], ax=ax2)
sns.kdeplot(scaled_df['PTS'], ax=ax2)
sns.kdeplot(scaled_df['FGM'], ax=ax2)
sns.kdeplot(scaled_df['FGA'], ax=ax2)
sns.kdeplot(scaled_df['FG%'], ax=ax2)
sns.kdeplot(scaled_df['3PM'], ax=ax2)
sns.kdeplot(scaled_df['3PA'], ax=ax2)
sns.kdeplot(scaled_df['3P%'], ax=ax2)
sns.kdeplot(scaled_df['FTM'], ax=ax2)
sns.kdeplot(scaled_df['FTA'], ax=ax2)
sns.kdeplot(scaled_df['FT%'], ax=ax2)
sns.kdeplot(scaled_df['OREB'], ax=ax2)
sns.kdeplot(scaled_df['DREB'], ax=ax2)
sns.kdeplot(scaled_df['REB'], ax=ax2)
sns.kdeplot(scaled_df['AST'], ax=ax2)
sns.kdeplot(scaled_df['STL'], ax=ax2)
sns.kdeplot(scaled_df['BLK'], ax=ax2)
sns.kdeplot(scaled_df['TOV'], ax=ax2)
sns.kdeplot(scaled_df['EFG%'], ax=ax2)
sns.kdeplot(scaled_df['TS%'], ax=ax2)
plt.show()


# In[19]:


scaled_df.head()


# In[20]:


scaled_df.describe()


# In[21]:


# Correlation matrix
scaled_df.corr()


# In[22]:


# Correlation plots
pd.scatter_matrix(scaled_df, figsize=(22,22))
plt.show()


# In[23]:


# Correlation heatmap
sns.set(rc={'figure.figsize':(80,10)})

corr = scaled_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);


# In[24]:


pca = PCA(n_components=5)
principalComponents = pca.fit_transform(scaled_df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2', 'PC3', 'PC4','PC5'])
principalDf.head()
#['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8','PC9', 'PC10','PC11','PC12',
#'PC13','PC14','PC15','PC16','PC17', 'PC18', 'PC19','PC20','PC21','PC22','PC23'])


# In[25]:


pca.explained_variance_ratio_


# In[26]:


data_1 = pd.DataFrame()


# In[27]:


data_1 = principalDf


# In[28]:


data_1.insert(5, "Picks", test['Total Picks'])


# In[29]:


data_1.insert(0, "Name", test['Name'])


# In[30]:


data_1 = data_1.astype({'Picks': 'int32'})


# In[31]:


y = data_1.iloc[:, 6]


# In[32]:


y.head()


# In[33]:


X = data_1.iloc[:, 1:6]


# In[34]:


X.head()


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[36]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[37]:


y_pred = regressor.predict(X_test)


# In[38]:


y_test.head()


# In[39]:


y_pred_1 = np.round(y_pred,0)


# In[40]:


y_pred_1 = y_pred_1.astype(int)


# In[41]:


y_pred_1.clip(0)


# In[42]:


from sklearn.metrics import confusion_matrix


# In[43]:


confusion_matrix(y_test, y_pred_1)


# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


accuracy_score(y_test, y_pred_1)


# In[46]:


# regression without PCA


# In[47]:


data2 = pd.DataFrame()


# In[48]:


data2 = scaled_df


# In[49]:


data2.columns


# In[50]:


data2.insert(21, "Picks", test['Total Picks'])


# In[51]:


data2.insert(0, "Names", test['Name'])


# In[52]:


data2.head()


# In[53]:


data2 = data2.astype({'Picks': 'int32'})


# In[54]:


X = data2.iloc[:, 1:22]


# In[55]:


y = data2.iloc[:, 22:]


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[58]:


from sklearn.linear_model import LinearRegression


# In[59]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[60]:


y_pred = regressor.predict(X_test)


# In[61]:


y_pred_2 = np.round(y_pred,0)


# In[62]:


y_pred_2 = y_pred_2.astype(int)


# In[63]:


y_pred_2.clip(0)


# In[65]:


confusion_matrix(y_test, y_pred_2)


# In[66]:


accuracy_score(y_test, y_pred_2)


# In[67]:


questions_data = pd.read_csv('questions_stats.csv')


# In[68]:


questions_data


# In[69]:


del questions_data['Names']


# In[70]:


questions_data.columns


# In[71]:


scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(questions_data[['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','EFG%','TS%']])
scaled_df = pd.DataFrame(scaled_df, columns=['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','EFG%','TS%'])


# In[72]:


scaled_df.head()


# In[73]:


y_pred = regressor.predict(scaled_df)


# In[74]:


y_pred


# In[75]:


y_pred = np.round(y_pred,0)


# In[76]:


y_pred = y_pred.astype(int)


# In[77]:


y_pred


# In[74]:


#XGBOOST


# In[86]:


from xgboost import XGBClassifier


# In[87]:


# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:





# In[88]:


# Predicting the Test set results
y_pred = classifier.predict(questions_data)


# In[89]:


# XGBOOST predictions are these
y_pred


# In[ ]:





######### time series analysis ########

#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA


# In[2]:


data = pd.read_csv('Seasons_Stats_1.csv')


# In[3]:


data.head()


# In[4]:


data = data.fillna(0)


# In[5]:


data = data.astype({'Picks': 'int32'})


# In[6]:


data = data.astype({'Year': 'int32'})


# In[7]:


data.head()


# In[8]:


len(data.columns)


# In[9]:


all_nba = pd.read_csv('all_nba.csv')


# In[10]:


all_nba.head()


# In[11]:


all_nba_1=pd.DataFrame()


# In[12]:


all_nba_1=all_nba


# In[13]:


all_nba_1 = all_nba_1.astype({'Year_end': 'int32'})


# In[14]:


all_nba_1['Name1'] = all_nba_1['Name1'].astype(str).str[:-2]
all_nba_1['Name2'] = all_nba_1['Name2'].astype(str).str[:-2]
all_nba_1['Name3'] = all_nba_1['Name3'].astype(str).str[:-2]
all_nba_1['Name4'] = all_nba_1['Name4'].astype(str).str[:-2]
all_nba_1['Name5'] = all_nba_1['Name5'].astype(str).str[:-2]


# In[15]:


all_nba_1.head()


# In[16]:


data.head()


# In[17]:


for index,row in data.iterrows():
    
    row = list(row)
    year = row[0]
    name = row[1]
   
    for index2,row2 in all_nba_1.iterrows():
    
        if (row2['Year_end'] == year) and (row2['Name1']== name or row2['Name2']== name or row2['Name3']== name or row2['Name4']== name or row2['Name5']== name):
                data.at[index,'Picks']=1


# In[ ]:


plt.plot(data.Picks)


# In[18]:


# verifying if the count matches
for index,row in data.iterrows():
    
    if (row['Picks'] >0) and (row['Player'] == "LeBron James") :
        print(row['Player'],row['Picks'])


# In[61]:


list_year = data['Year'].unique()


# In[62]:


list_year


# In[137]:


data.head()


# In[215]:


test_df = pd.DataFrame()


# In[216]:


test1_df = pd.DataFrame()


# In[234]:


test1_df = data[data['Player'] == 'Karl-Anthony Towns']


# In[235]:


test1_df


# In[246]:


test1_df = test1_df.append(data[data['Player'] == 'Stephen Curry'])


# In[ ]:


test1_df = test1_df.append(data[data['Player'] == 'Kyrie Irving'])


# In[247]:


test1_df


# In[248]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[249]:


ax = sns.lineplot(x="Year", y="PTS", hue='Player',data=test1_df)


# In[145]:


len(data.Player.unique())


# In[169]:


# Correlation heatmap
sns.set(rc={'figure.figsize':(80,10)})

corr = test1_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);


# In[ ]:
