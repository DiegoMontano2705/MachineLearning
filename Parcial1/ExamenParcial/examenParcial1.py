import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

def applySciKitTools(x,y,x_test,y_test):
  #Usando Regresion logistica
  model = LogisticRegression()
  model.fit(x,y)
  accuracy = model.score(x_test,y_test)
  print("Precision Regresion Lineal Scikit-Learn:",accuracy)

  #Usando arboles de decision
  model = tree.DecisionTreeClassifier(max_depth=2, random_state=30)
  model.fit(x,y)
  accuracy= model.score(x_test, y_test)
  print("Precision Arboles de Decision Scikit-Learn",accuracy)

  #Usando 50-NearestNeighbors
  model = KNeighborsClassifier(n_neighbors=50, algorithm='brute')
  model.fit(x, y)
  accuracy = model.score(x_test, y_test)
  print("Precision 50-NearestNeighbors Scikit-Learn:",accuracy)

def normalize(x,y):
  m1,n1 = x.shape
  aux = np.ones(m1)
  aux = aux.reshape(m1,1)
  x = np.append(aux,x,axis=1)
  y = y.reshape(-1,1)
  return x,y

def GlassDS(file):
  # Leyendo dataset
  data = pd.read_csv(file,delimiter=' ')
  data = data.drop(columns=['ID'],axis=1)

  x = data[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']] #features del vector de entrada
  y = data['Type'] #Categoria a ser predecida

  
  # Creando vector de entrenamiento
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=40)
  x_train = x_train.values  
  y_train = y_train.values  
  y_test = y_test.values

  #Normalizando
  x_train,y_train = normalize(x_train,y_train)
  x_test,y_test = normalize(x_test,y_test)


  #Aplicando la modelacion
  print("Glass Identification Database")
  applySciKitTools(x_train,y_train,x_test,y_test)

def IonosphereDS(file):
  # Leyendo dataset
  data = pd.read_csv(file,delimiter=' ')
  data = data.drop(columns=['ID'],axis=1)
  Class = {'good':1,'bad':0}
  data.Class = [Class[item] for item in data.Class]

  x = data[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34']] #features del vector de entrada
  y = data['Class'] #Categoria a ser predecida

  # Creando vector de entrenamiento
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=40)
  x_train = x_train.values  
  y_train = y_train.values  
  y_test = y_test.values

  #Normalizando
  x_train,y_train = normalize(x_train,y_train)
  x_test,y_test = normalize(x_test,y_test)


  #Aplicando la modelacion
  print("Johns Hopkins University Ionosphere database")
  applySciKitTools(x_train,y_train,x_test,y_test)



def main():
  GlassDS("Glass.txt")
  IonosphereDS("Ionosphere.txt")


if __name__ == '__main__':
  main()