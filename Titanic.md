```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, SMOTENC
import IPython
import os
```


```python
for dirname, _, filenames in os.walk('Desktop/AD/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    Desktop/AD/gender_submission.csv
    Desktop/AD/test.csv
    Desktop/AD/train.csv
    


```python
df = pd.read_csv("Desktop/AD/train.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
sns.heatmap(df.isnull(), cbar=False)
```




    <AxesSubplot:>




    
![png](output_3_1.png)
    



```python
sns.countplot(x='Survived', data=df)
```




    <AxesSubplot:xlabel='Survived', ylabel='count'>




    
![png](output_4_1.png)
    



```python
sns.countplot(x='Sex', data=df)
```




    <AxesSubplot:xlabel='Sex', ylabel='count'>




    
![png](output_5_1.png)
    



```python
sns.countplot(x='Survived', hue='Sex', data=df)
```




    <AxesSubplot:xlabel='Survived', ylabel='count'>




    
![png](output_6_1.png)
    



```python
sns.countplot(x='Survived', hue='Pclass', data=df)
```




    <AxesSubplot:xlabel='Survived', ylabel='count'>




    
![png](output_7_1.png)
    



```python
plt.hist(df['Age'].dropna())
```




    (array([ 54.,  46., 177., 169., 118.,  70.,  45.,  24.,   9.,   2.]),
     array([ 0.42 ,  8.378, 16.336, 24.294, 32.252, 40.21 , 48.168, 56.126,
            64.084, 72.042, 80.   ]),
     <BarContainer object of 10 artists>)




    
![png](output_8_1.png)
    



```python
sns.boxplot(df['Sex'], df['Age'])
```

    D:\Programs\A\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='Sex', ylabel='Age'>




    
![png](output_9_2.png)
    



```python
def impute_missing_age(columns):

    age = columns[0]

    passenger_class = columns[1]

    

    if pd.isnull(age):

        if(passenger_class == 1):

            return df[df['Pclass'] == 1]['Age'].mean()

        elif(passenger_class == 2):

            return df[df['Pclass'] == 2]['Age'].mean()

        elif(passenger_class == 3):

            return df[df['Pclass'] == 3]['Age'].mean()

        

    else:

        return age

df['Age'] = df[['Age', 'Pclass']].apply(impute_missing_age, axis = 1)
sns.heatmap(df.isnull(), cbar=False)
```




    <AxesSubplot:>




    
![png](output_10_1.png)
    



```python
df.drop('Cabin', axis=1, inplace = True)
df.dropna(inplace = True)
sex_data = pd.get_dummies(df['Sex'], drop_first = True)
embarked_data = pd.get_dummies(['Embarked'], drop_first = True)
df = pd.concat([df, sex_data, embarked_data], axis = 1)
df.drop(['Name', 'PassengerId', 'Ticket', 'Sex', 'Embarked'], axis = 1, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_data = df['Survived']
x_data = df.drop('Survived', axis = 1)
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)
from sklearn.metrics import classification_report
print(classification_report(y_test_data, predictions))
```

                  precision    recall  f1-score   support
    
               0       0.76      0.89      0.82       106
               1       0.78      0.59      0.67        73
    
        accuracy                           0.77       179
       macro avg       0.77      0.74      0.74       179
    weighted avg       0.77      0.77      0.76       179
    
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test_data, predictions))
```

    [[94 12]
     [30 43]]
    


```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)
        
df['Survived'] = df.Survived.astype("category").cat.codes

label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])   
```


```python
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
from sklearn.tree import DecisionTreeClassifier
tr = DecisionTreeClassifier(random_state= 42)
tr.fit(X_train, y_train) 
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    




    DecisionTreeClassifier(random_state=42)




```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```


```python
tree_clf = DecisionTreeClassifier(random_state=100)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    ================================================
    Accuracy Score: 98.07%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.972637    0.995475  0.980738    0.984056      0.981105
    recall       0.997449    0.952381  0.980738    0.974915      0.980738
    f1-score     0.984887    0.973451  0.980738    0.979169      0.980647
    support    392.000000  231.000000  0.980738  623.000000    623.000000
    _______________________________________________
    Confusion Matrix: 
     [[391   1]
     [ 11 220]]
    
    Test Result:
    ================================================
    Accuracy Score: 76.49%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.786585    0.730769  0.764925    0.758677      0.763467
    recall       0.821656    0.684685  0.764925    0.753170      0.764925
    f1-score     0.803738    0.706977  0.764925    0.755358      0.763662
    support    157.000000  111.000000  0.764925  268.000000    268.000000
    _______________________________________________
    Confusion Matrix: 
     [[129  28]
     [ 35  76]]
    
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 4332 candidates, totalling 12996 fits
    Best paramters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 13, 'min_samples_split': 2, 'splitter': 'random'})
    Train Result:
    ================================================
    Accuracy Score: 83.47%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.814815    0.890244  0.834671    0.852529      0.842783
    recall       0.954082    0.632035  0.834671    0.793058      0.834671
    f1-score     0.878966    0.739241  0.834671    0.809103      0.827158
    support    392.000000  231.000000  0.834671  623.000000    623.000000
    _______________________________________________
    Confusion Matrix: 
     [[374  18]
     [ 85 146]]
    
    Test Result:
    ================================================
    Accuracy Score: 78.73%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.763158    0.846154  0.787313    0.804656      0.797533
    recall       0.923567    0.594595  0.787313    0.759081      0.787313
    f1-score     0.835735    0.698413  0.787313    0.767074      0.778859
    support    157.000000  111.000000  0.787313  268.000000    268.000000
    _______________________________________________
    Confusion Matrix: 
     [[145  12]
     [ 45  66]]
    
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Train Result:
    ================================================
    Accuracy Score: 98.07%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.977387    0.986667  0.980738    0.982027      0.980828
    recall       0.992347    0.961039  0.980738    0.976693      0.980738
    f1-score     0.984810    0.973684  0.980738    0.979247      0.980685
    support    392.000000  231.000000  0.980738  623.000000    623.000000
    _______________________________________________
    Confusion Matrix: 
     [[389   3]
     [  9 222]]
    
    Test Result:
    ================================================
    Accuracy Score: 79.48%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.800000    0.785714  0.794776    0.792857      0.794083
    recall       0.866242    0.693694  0.794776    0.779968      0.794776
    f1-score     0.831804    0.736842  0.794776    0.784323      0.792473
    support    157.000000  111.000000  0.794776  268.000000    268.000000
    _______________________________________________
    Confusion Matrix: 
     [[136  21]
     [ 34  77]]
    
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = RandomizedSearchCV(estimator=rf_clf, scoring='f1',param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)

rf_cv.fit(X_train, y_train)
rf_best_params = rf_cv.best_params_
print(f"Best paramters: {rf_best_params})")

rf_clf = RandomForestClassifier(**rf_best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Best paramters: {'n_estimators': 1400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True})
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Train Result:
    ================================================
    Accuracy Score: 89.57%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.877598    0.936842  0.895666    0.907220      0.899565
    recall       0.969388    0.770563  0.895666    0.869975      0.895666
    f1-score     0.921212    0.845606  0.895666    0.883409      0.893178
    support    392.000000  231.000000  0.895666  623.000000    623.000000
    _______________________________________________
    Confusion Matrix: 
     [[380  12]
     [ 53 178]]
    
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Test Result:
    ================================================
    Accuracy Score: 81.34%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.792350    0.858824  0.813433    0.825587      0.819882
    recall       0.923567    0.657658  0.813433    0.790612      0.813433
    f1-score     0.852941    0.744898  0.813433    0.798920      0.808192
    support    157.000000  111.000000  0.813433  268.000000    268.000000
    _______________________________________________
    Confusion Matrix: 
     [[145  12]
     [ 38  73]]
    
    


```python
n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 10]
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = GridSearchCV(rf_clf, params_grid, scoring="f1", cv=3, verbose=2, n_jobs=-1)


rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
```

    Fitting 3 folds for each of 768 candidates, totalling 2304 fits
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Best parameters: {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    Train Result:
    ================================================
    Accuracy Score: 89.41%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.875576    0.936508  0.894061    0.906042      0.898169
    recall       0.969388    0.766234  0.894061    0.867811      0.894061
    f1-score     0.920097    0.842857  0.894061    0.881477      0.891457
    support    392.000000  231.000000  0.894061  623.000000    623.000000
    _______________________________________________
    Confusion Matrix: 
     [[380  12]
     [ 54 177]]
    
    Test Result:
    ================================================
    Accuracy Score: 82.09%
    _______________________________________________
    CLASSIFICATION REPORT:
                        0           1  accuracy   macro avg  weighted avg
    precision    0.797814    0.870588  0.820896    0.834201      0.827956
    recall       0.929936    0.666667  0.820896    0.798301      0.820896
    f1-score     0.858824    0.755102  0.820896    0.806963      0.815864
    support    157.000000  111.000000  0.820896  268.000000    268.000000
    _______________________________________________
    Confusion Matrix: 
     [[146  11]
     [ 37  74]]
    
    


```python
print (str(tr.score(X_train, y_train) * 100 ) + ' %')
```

    98.07383627608347 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
y_pred = tr.predict(X_test)
print (str(accuracy_score(y_pred, y_test) * 100 ) + ' %')
```

    77.23880597014924 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
tr = SVC(C = 1.0, kernel = 'linear')
tr.fit(X_train, y_train)
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    




    SVC(kernel='linear')




```python
y_pred = tr.predict(X_test)

print (str(accuracy_score(y_pred, y_test)*100 ) + ' %')
```

    79.1044776119403 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    




    LogisticRegression()




```python
y_pred = lr.predict(X_test)
print (str(accuracy_score(y_pred, y_test)*100) + ' %')
```

    79.8507462686567 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
ac = neighbors.KNeighborsClassifier(n_neighbors=10)
ac.fit(X_train, y_train) 
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    




    KNeighborsClassifier(n_neighbors=10)




```python
y_pred = ac.predict(X_test)
print (str(accuracy_score(y_pred, y_test)*100) + ' %')
```

    67.16417910447761 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    


```python
rf = RandomForestClassifier(n_estimators=2)
rf.fit(X_train, y_train)
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    




    RandomForestClassifier(n_estimators=2)




```python
y_pred = rf.predict(X_test)
print (str(accuracy_score(y_pred, y_test)*100) + ' %')
```

    78.35820895522389 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
score = cross_val_score(tr, X, y, cv = 10)

print (str(score.mean()*100) + ' %')
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    78.6729088639201 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


```python
sns.heatmap(confusion_matrix(rf.predict(X_test), y_test),annot=True)
print (str(f1_score(tr.predict(X_test), y_test)*100) + ' %')
```

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    

    73.58490566037736 %
    

    D:\Programs\A\lib\site-packages\sklearn\utils\validation.py:1688: FutureWarning: Feature names only support names that are all strings. Got feature names with dtypes: ['int', 'str']. An error will be raised in 1.2.
      warnings.warn(
    


    
![png](output_34_3.png)
    



```python

```
