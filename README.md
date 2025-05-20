# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model. 9.End the Program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shasmithaa Sankar
RegisterNumber:  212224040311
*/
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/d84b19d9-0063-4367-b3ec-9c6b9710ecdd)
![image](https://github.com/user-attachments/assets/cb75577c-3ef5-4a21-b846-4fe539c8cb9e)
![image](https://github.com/user-attachments/assets/39c5e1d9-9fee-40fb-a8ad-15b981c19bfa)
![image](https://github.com/user-attachments/assets/b16330cb-274e-4d0c-ba17-458b26989d8d)
![image](https://github.com/user-attachments/assets/a3964aba-6af7-4db6-b2ed-5f4e98bac337)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
