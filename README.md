# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

from google.colab import drive
drive.mount('/content/drive')

<img width="174" alt="Screenshot 2024-12-11 193948" src="https://github.com/user-attachments/assets/ad49162f-9be9-4880-85bd-342852c01e02" />


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from google.colab import files
uploaded = files.upload()

# Access the uploaded file using the correct filename
# assuming the uploaded filename is 'income(1) (1) (1) (2).csv'
filename = list(uploaded.keys())[0] # Get the filename from the uploaded dictionary

data = pd.read_csv(filename, na_values=[" ?"])  # Use the actual filename from the upload process
<img width="431" alt="Screenshot 2024-12-11 194257" src="https://github.com/user-attachments/assets/4560f14b-85d3-419a-8661-b9e3a39d9870" />

data
<img width="886" alt="Screenshot 2024-12-11 193931" src="https://github.com/user-attachments/assets/d818b12d-f428-47b7-9c8b-b606ad609456" />


data.isnull().sum()
<img width="125" alt="Screenshot 2024-12-11 193829" src="https://github.com/user-attachments/assets/9824aa6e-3be3-4c66-bdda-4ecbc958612a" />

missing = data[data.isnull().any(axis=1)]
missing
<img width="887" alt="Screenshot 2024-12-11 193804" src="https://github.com/user-attachments/assets/0bac2c1b-d4db-4096-ae58-b2409915d590" />


data2 = data.dropna(axis=0)
data2
<img width="881" alt="Screenshot 2024-12-11 193743" src="https://github.com/user-attachments/assets/c66d6229-ab56-4824-b2fb-8dde0e3ff004" />


sal=data['SalStat']

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

<img width="839" alt="Screenshot 2024-12-11 193711" src="https://github.com/user-attachments/assets/5a8a1123-f0e3-46f4-9eeb-be7d51132724" />


sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs

<img width="268" alt="Screenshot 2024-12-11 193642" src="https://github.com/user-attachments/assets/ccbb193f-f5e4-49f1-9282-707312d66975" />

data2

<img width="868" alt="Screenshot 2024-12-11 193609" src="https://github.com/user-attachments/assets/1d46646b-6e2d-4e21-9088-48836448baf1" />

new_data = pd.get_dummies(data2, drop_first=True)
new_data

<img width="884" alt="Screenshot 2024-12-11 193546" src="https://github.com/user-attachments/assets/02990e90-47d7-4b88-a7d9-536d05f84395" />

columns_list = list(new_data.columns)
print(columns_list)

<img width="452" alt="Screenshot 2024-12-11 193420" src="https://github.com/user-attachments/assets/5636b0e6-c8fd-4daf-8cb2-1c54d85b5107" />


['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'SalStat', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']



features = list(set(columns_list)-set(['SalStat']))
print(features)

<img width="449" alt="Screenshot 2024-12-11 193336" src="https://github.com/user-attachments/assets/19fdcdfd-c036-44a0-9770-152f730641ec" />


['EdType_ Doctorate', 'relationship_ Not-in-family', 'nativecountry_ England', 'race_ Asian-Pac-Islander', 'nativecountry_ Mexico', 'EdType_ 7th-8th', 'capitalgain', 'nativecountry_ India', 'occupation_ Transport-moving', 'nativecountry_ Laos', 'relationship_ Other-relative', 'occupation_ Farming-fishing', 'capitalloss', 'maritalstatus_ Widowed', 'relationship_ Unmarried', 'EdType_ 1st-4th', 'nativecountry_ Peru', 'race_ Black', 'nativecountry_ Thailand', 'nativecountry_ Iran', 'nativecountry_ Nicaragua', 'EdType_ Some-college', 'EdType_ 9th', 'occupation_ Tech-support', 'maritalstatus_ Married-civ-spouse', 'nativecountry_ Greece', 'nativecountry_ Holand-Netherlands', 'race_ White', 'nativecountry_ Taiwan', 'EdType_ 5th-6th', 'occupation_ Craft-repair', 'EdType_ 12th', 'nativecountry_ Hungary', 'occupation_ Protective-serv', 'hoursperweek', 'EdType_ Prof-school', 'nativecountry_ China', 'JobType_ Private', 'occupation_ Machine-op-inspct', 'nativecountry_ El-Salvador', 'EdType_ Masters', 'occupation_ Other-service', 'relationship_ Wife', 'JobType_ Self-emp-inc', 'EdType_ Preschool', 'nativecountry_ Poland', 'JobType_ State-gov', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'occupation_ Handlers-cleaners', 'nativecountry_ Ireland', 'nativecountry_ Germany', 'JobType_ Local-gov', 'nativecountry_ United-States', 'nativecountry_ Cuba', 'nativecountry_ Yugoslavia', 'nativecountry_ South', 'occupation_ Prof-specialty', 'nativecountry_ Jamaica', 'EdType_ Bachelors', 'nativecountry_ Canada', 'nativecountry_ Hong', 'nativecountry_ Scotland', 'JobType_ Without-pay', 'nativecountry_ Portugal', 'occupation_ Exec-managerial', 'nativecountry_ Vietnam', 'nativecountry_ Trinadad&Tobago', 'EdType_ Assoc-voc', 'nativecountry_ Honduras', 'nativecountry_ Ecuador', 'relationship_ Own-child', 'nativecountry_ Guatemala', 'race_ Other', 'nativecountry_ Haiti', 'maritalstatus_ Separated', 'nativecountry_ Japan', 'occupation_ Armed-Forces', 'nativecountry_ Philippines', 'maritalstatus_ Married-spouse-absent', 'occupation_ Sales', 'EdType_ 11th', 'nativecountry_ Puerto-Rico', 'maritalstatus_ Never-married', 'nativecountry_ Dominican-Republic', 'gender_ Male', 'nativecountry_ France', 'EdType_ Assoc-acdm', 'nativecountry_ Italy', 'occupation_ Priv-house-serv', 'age', 'JobType_ Self-emp-not-inc', 'EdType_ HS-grad', 'nativecountry_ Columbia', 'maritalstatus_ Married-AF-spouse']



y = new_data['SalStat'].values
print(y)
<img width="112" alt="Screenshot 2024-12-11 193215" src="https://github.com/user-attachments/assets/eb692f74-4be4-443f-adfa-6b8558457193" />


x = new_data[features].values
print(x)

<img width="252" alt="Screenshot 2024-12-11 193156" src="https://github.com/user-attachments/assets/3abd7937-ed20-4045-9427-5e5ebdf50e56" />


x=new_data[features].values
print(x)

<img width="248" alt="Screenshot 2024-12-11 193144" src="https://github.com/user-attachments/assets/3df2d457-6c24-4348-8628-b409ae888544" />


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

KNN_classifier = KNeighborsClassifier(n_neighbors=5)

KNN_classifier.fit(train_x,train_y)

<img width="196" alt="Screenshot 2024-12-11 193128" src="https://github.com/user-attachments/assets/260e8538-d1da-49cc-aa92-72dd8b32c555" />


prediction = KNN_classifier.predict(test_x)

confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)

<img width="95" alt="Screenshot 2024-12-11 193107" src="https://github.com/user-attachments/assets/38236a6d-309d-4440-8ff2-a03c37ee1824" />


accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

<img width="116" alt="Screenshot 2024-12-11 193047" src="https://github.com/user-attachments/assets/4f865de2-3d14-4405-9ae3-58b70b4c19f0" />


print('Misclassified samples: %d' % (test_y !=prediction).sum())

<img width="161" alt="Screenshot 2024-12-11 193015" src="https://github.com/user-attachments/assets/cb8d2b56-37a7-4716-99ff-6729fbcd5b22" />

data.shape

<img width="73" alt="Screenshot 2024-12-11 192929" src="https://github.com/user-attachments/assets/53f87f0a-c6c8-4acc-ae1b-6acf6a4926ab" />

















# RESULT:
       # INCLUDE YOUR RESULT HERE
