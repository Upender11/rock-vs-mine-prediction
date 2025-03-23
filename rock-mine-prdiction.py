
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



sonar_data=pd.read_csv('/sonar data.csv',header=None)
sonar_data.head()

sonar_data.shape

sonar_data.describe() #describes the statistical data

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

a=sonar_data.drop(columns=60,axis=1)
b=sonar_data[60]

print(a)
print(b)



X_train,X_test,Y_train,Y_test=train_test_split(a,b,test_size=0.1,stratify=b,random_state=1)

print(a.shape,X_train.shape,X_test.shape)

print(X_train)
print(Y_train)



model=LogisticRegression()

model.fit(X_train,Y_train)


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print("Accuracy on the training data: ",training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print("Accuracy on the test data: ",test_data_accuracy)


input_data=(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)

input_data_as_numpy_array=np.asarray(input_data)


input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
  print("The Object Is A Rock")
else:
  print("The Object Is A Mine")

