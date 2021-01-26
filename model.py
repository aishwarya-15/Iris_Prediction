import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pickle

df = pd.read_csv('C:\Users\ELCOT\Downloads\datasets_19_420_Iris.csv')

X = df.drop(['Id','Species'],axis=1).astype(float)
y = df['Species']
sv = SVC(kernel='linear')
sv.fit(X,y)
pickle.dump(sv, open('model.pkl', 'wb'))