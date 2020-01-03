#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import RandomForestClassifier
plt.style.use('default') 


#%% Load Data and transform
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')
# fml_count
train_data['fmly_count'] = train_data['SibSp'] + train_data['Parch']
test_data['fmly_count'] = test_data['SibSp'] + test_data['Parch']
# suffix
train_data['Name'] = train_data['Name'].str.extract(r'(\s[a-zA-Z]+\.)')
test_data['Name'] = test_data['Name'].str.extract(r'(\s[a-zA-Z]+\.)')

train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# train_data[train_data['Cabin']!=0] = train_data[train_data['Cabin']!=0][:1]
print(train_data)



# %% (x,y) = (SibSp, Parch) for died, survived
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
for survived in np.arange(2):
    plt_data = train_data[train_data['Survived'] == survived]
    for xxx, p in plt_data.iterrows():
        noise = np.random.uniform(size=2, low=-.5, high=.5)
        ax[survived].scatter(p['SibSp'] + noise[0], p['Parch'] + noise[1], c='red', alpha = 0.2)

plt.show()

# %% Remove all but suffix from names
train_data['Name'] = train_data['Name'].str.extract(r'(\s[a-zA-Z]+\.)')
test_data['Name'] = test_data['Name'].str.extract(r'(\s[a-zA-Z]+\.)')
temp = train_data.copy()
total = temp['Name'].value_counts()
print('Total counts:\n', total)
temp = train_data.copy()
temp = temp[temp.Survived == 0]
died = temp['Name'].value_counts()
print('\nDead Counts:\n', died)
temp = train_data.copy()
temp = temp[temp.Survived == 1]
survived = temp['Name'].value_counts()
print('\nSurvived Counts:\n', survived)
percents = survived / total
percents.fillna(0, inplace=True)
print('\nPercents\n', percents)



# %% Look at age distribution (0 for no age given. Lots of 0s...)
temp = train_data.copy()
temp['Age'].fillna(0, inplace=True)
sns.distplot(a=temp[temp['Survived']==0]['Age'])
sns.distplot(a=temp[temp['Survived']==1]['Age'])
plt.show()

# %% Make feature fmly_count



# %%
y = train_data['Survived']

features = ['Pclass', 'fmly_count', 'Fare', 'Name', 'Sex', 'Age', 'label']
train_data['label'] = 0
test_data['label'] = 1
concat_df = pd.concat([train_data, test_data])
concat_X = pd.get_dummies(concat_df[features])
# print('printing concat_X\n', concat_X)
X = concat_X[concat_X['label'] == 0]
X_test = concat_X[concat_X['label'] == 1]
# print('Printing X\n', X)
# print('Printin X_test\n', X_test)
X.drop(columns=['label'])
X_test.drop(columns=['label'])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# %%
