import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats

# 教育用、検証用データの読み込み
train_df = pd.read_csv('./8010_titanic_data/train.csv')
test_df = pd.read_csv('./8010_titanic_data/test.csv')

combine = [train_df, test_df]

# 'Ticket' 'Cabin'カラムを削除
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

combine = [train_df, test_df]

# 'Name'カラムから敬称を'Title'カラムとして新たに抽出
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# 'Title'カラムを主要な敬称に置き換える
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# 'Title'カラムの文字列を数値に置き換え
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

# 教育用データから'Name' 'PassengerId'カラムを削除、検証用データから'Name'を削除
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

# 'Sex'カラムの文字列を数値に置き換える
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# 'Sex' 'Pclass'ごとに'Age'の中央値を求め、それぞれのNANを各中央値で埋める（計６パターン）
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# 'Age' カラムを５つに分類する新カラム'AgeBand'を作成    
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

# 'AgeBand'で確認した４つの閾値に沿って'Age'を5つの数値に置き換える
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# 'AgeBand'を削除    
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

# 'SibSp' 'Parch'から家族の構成人数を表す新カラム'FamilySize'を作成（自分を含む）
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 一緒に乗船してる家族がいない場合は、新カラム'IsAlone'に「1」を代入、家族がいる場合は「0」
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 'Parch' 'SibSp' 'FmailySize'のカラムを削除
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

# 'Age'に'Pclass'を重みづけた新カラム'Age*Class'を作成
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# 'Embarked'のNANを自身カラム内の最頻値で埋める
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 'Embarked'内の文字列を数値に置き換える
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# 検証用データのNANを最頻値で埋める
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# 教育用データに'Fare'を４つに分類する新カラム'FareBand'を作成
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

# 'FareBand'で確認した3つの閾値に沿って'Fare'を4つの数値に置き換える
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# 教育用データの'FareBand'カラムを削除
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# 教育用データを説明変数と目的変数に分けて格納する
train_y = train_df['Survived']
train_X = train_df.drop(['Survived'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)
#print(train_X.head())
#print(train_y.head())
#print()
#print(test_df.head())


# MAXの正解率におけるモデル名とパラメータの格納先を定義
max_score = 0
score_list = []
best_model = None
best_param = None

# ランダムサーチする機械学習モデルとそのパラメータを設定
models = [LogisticRegression(), LinearSVC(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
params = [
    {'C': [10**i for i in range(-5, 5)], 
    'random_state': scipy.stats.randint(1, 101)}, 
    {'C': [10**i for i in range(-5, 5)], 
    'multi_class': ['ovr', 'crammer_singer'], 
    'random_state': scipy.stats.randint(1, 101)}, 
    {'C': [10**i for i in range(-5, 5)], 
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
    'decision_function_shape': ['ovo', 'ovr'], 
    'random_state': scipy.stats.randint(1, 101)}, 
    {'max_depth': scipy.stats.randint(1, 11), 
    'random_state': scipy.stats.randint(1, 101)}, 
    {'max_depth': scipy.stats.randint(1, 11), 
    'random_state': scipy.stats.randint(1, 101)}, 
    {'n_neighbors': [i for i in range(1, 11)]}
    ]

# ランダムサーチでF値の高いパラメータを探索
for model, param in zip(models, params):
    clf = RandomizedSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_df = clf.predict(test_df)
    score = clf.score(train_X, train_y)
    score_list.append(score)

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__
        best_pred = pred_df

# 教育用データにおいてMAXの正解率を出す学習モデルを表示
print('学習モデル:{}, \nパラメータ:{}'.format(best_model, best_param))
print('MAX_Accuracy:', round(max_score*100, 2))

for model, score in zip(models, score_list):
    print('{}: {}%'.format(model.__class__.__name__, round(score*100, 2)))
