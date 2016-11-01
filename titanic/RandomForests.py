#Index([u'Sex', u'Title', u'Age*Fare', u'Age+Pclass', u'Age+SibSp', u'Age-Names', u'Fare-Age', u'Fare-Pclass', u'Fare+Names', u'Fare+CabinLetter', u'Pclass-Fare', u'Names-Age', u'Names+CabinLetter'], dtype='object')
# coding: utf-8

# In[128]:

import numpy as np
import pandas as pd


# In[159]:

initial_train_df = pd.read_csv("train.csv")




selected_features_df = initial_train_df[['Survived','Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked']]
#missing_age_rows = float(len(selected_features_df[pd.isnull(selected_features_df['Age'])]))
#len(selected_features_df)
# selected_features_df = selected_features_df.dropna(subset=['Age'],how='any',axis = 0)
median_age = selected_features_df['Age'].dropna().median()
mode_embarked = selected_features_df['Embarked'].dropna().mode().values
# print mode_embarked
selected_features_df.loc[ (selected_features_df.Age.isnull()), 'Age'] = median_age
selected_features_df.loc[ (selected_features_df.Embarked.isnull()), 'Embarked'] = mode_embarked






Ports = list(enumerate(np.unique(selected_features_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
selected_features_df.Embarked = selected_features_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

selected_features_df


# 
# **One hot encoding**

# In[195]:

selected_features_df['Gender'] = selected_features_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#df_train = pd.get_dummies(selected_features_df, columns = ['Sex'])
selected_features_df = selected_features_df.drop(['Sex'],axis=1)
selected_features_df
df_train = selected_features_df


# In[196]:

predictors = list(df_train.columns)
predictors.remove('Survived')
print predictors
x = df_train[predictors].values
y = list(df_train['Survived'].values)


# In[166]:

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

# max_score = 0
# max_estimators = 0
# for i in range(100,501,10):
#     alg = RandomForestClassifier(n_estimators = i, max_features = None, warm_start = True)
#     scores = cross_validation.cross_val_score(alg,x,y,cv=10)
#     print i,"The cross validation accuracy is ", scores.mean()
#     if scores.mean() > max_score:
#         max_score = scores.mean()
#         max_estimators = i
# max_estimators


# In[143]:

# test_df.columns
# max_estimators


# In[197]:

test_df = pd.read_csv("test.csv")
p_ids = test_df['PassengerId']
test_df = test_df[['Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked']]

test_median_age = test_df['Age'].dropna().median()
test_mode_embarked = test_df['Embarked'].dropna().mode().values


test_df.loc[ (test_df.Age.isnull()), 'Age'] = test_median_age
test_df.loc[ (test_df.Embarked.isnull()), 'Embarked'] = test_mode_embarked


test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int) 
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df = test_df.drop(['Sex'],axis=1)

#test_df = pd.get_dummies(test_df, columns = ['Sex'])
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
print test_df.columns[pd.isnull(test_df).any()]


# In[198]:

alg = RandomForestClassifier(n_estimators = 140)
alg.fit(x,y)
survived = alg.predict(test_df).astype(int)
# test_pred


# In[199]:

# survived = []
# for pred in test_pred:
#     if pred[0] >= pred[1]:
#         survived.append(0)
#     else:
#         survived.append(1)
# len(survived)


# In[200]:
print survived
survived = np.asanyarray(survived)
ans = pd.DataFrame(data = p_ids)
ans["Survived"] = survived
ans.to_csv("rfmodel7.csv",index=False)


# In[ ]:



