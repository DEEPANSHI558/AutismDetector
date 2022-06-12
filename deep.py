import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

data_set=pd.read_csv('autism_screening.csv')
data_set.head()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X=data_set.iloc[:,0:10]
y=data_set.iloc[:,-7]
imputer.fit(X.iloc[:, 0:10])
X.iloc[:, 0:10] = imputer.transform(X.iloc[:, 0:10])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

# from sklearn.svm import SVC
# clf1 = SVC(kernel = 'rbf', random_state = 3,C=5)  
# clf1.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)

pickle.dump(dt,open('deep.pkl','wb'))
