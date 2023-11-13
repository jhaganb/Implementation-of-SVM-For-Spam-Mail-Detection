# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jhagan B
RegisterNumber:  212220040066
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

Result Output:

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/928793f5-16b4-4300-bf83-140ef7db349f)

data.head():

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/c18ad42a-712e-4337-bc3c-13250d9b8f91)

data.info():

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/e39f9a79-bbbc-420c-a156-0e6bf90b7b88)

data.isnull().sum():

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/b75b268c-b24c-4433-a91f-ce4b1fd4eb8a)

Y_prediction value:

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/323a61d5-5b09-45b6-b09c-34f2f61a2432)

Accuracy value:

![image](https://github.com/jhaganb/Implementation-of-SVM-For-Spam-Mail-Detection/assets/63654882/98b66ee0-9e64-4905-ada9-ca9cfa845a2e)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
