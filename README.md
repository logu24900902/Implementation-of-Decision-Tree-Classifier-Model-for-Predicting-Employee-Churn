# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithem
### 1.Import the required libraries.

### 2.Upload and read the dataset.

### 3.Check for any null values using the isnull() function.

### 4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

### 5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.thm


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Logu R
RegisterNumber:  212224230141
*/
```
```Python
  import pandas as pd
  data=pd.read_csv("Employee.csv")
  print("data.head():")
  data.head()

  
  print("data.info():")
  data.info()

  print("isnull() and sum():")
  data.isnull().sum()

  print("data value counts():")
  data["left"].value_counts()

  from sklearn.preprocessing import LabelEncoder
  le=LabelEncoder()

  print("data.head() for Salary:")
  data["salary"]=le.fit_transform(data["salary"])
  data.head()

  print("x.head():")
  x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
  x.head()

  y=data["left"]
  from sklearn.model_selection import train_test_split
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
  from sklearn.tree import DecisionTreeClassifier
  dt=DecisionTreeClassifier(criterion="entropy")
  dt.fit(x_train,y_train)
  y_pred=dt.predict(x_test)

  print("Accuracy value:")
  from sklearn import metrics
  accuracy=metrics.accuracy_score(y_test,y_pred)
  accuracy

  print("Data Prediction:")
  dt.predict([[0.5,0.8,9,260,6,0,1,2]])

  from sklearn.tree import plot_tree
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(8,6))
  plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
  plt.show()
```

## Output:
<img width="1031" height="183" alt="image" src="https://github.com/user-attachments/assets/a0def375-dbfa-43b6-bbbc-b2653fddfced" />


<img width="1024" height="378" alt="image" src="https://github.com/user-attachments/assets/7d499b27-60e6-4838-8c9f-126c96aa9f51" />


<img width="1023" height="311" alt="image" src="https://github.com/user-attachments/assets/41e5d36d-3037-44bf-ae3d-7d21136299e6" />

<img width="609" height="127" alt="image" src="https://github.com/user-attachments/assets/2feb5a1c-830b-4fd7-87da-9d412da9a1d7" />

<img width="1023" height="175" alt="image" src="https://github.com/user-attachments/assets/6283504c-e5af-4a24-9237-32d840b96186" />

<img width="995" height="177" alt="image" src="https://github.com/user-attachments/assets/78974813-669a-439c-a240-441e74f83877" />


<img width="808" height="629" alt="image" src="https://github.com/user-attachments/assets/5abd5022-4c59-485a-b567-b8155c23010b" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
