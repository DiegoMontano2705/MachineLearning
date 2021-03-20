import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.impute import KNNImputer


def readFile():
  url = "https://raw.githubusercontent.com/DiegoMontano2705/MachineLearning/main/Parcial1/RegresionLogistica/Default.txt"
  data = pd.read_csv(url, sep='\t')
  default = {'Yes': 0, 'No': 1}
  student = {'Yes': 0, 'No': 1}
  data.default = [default[item] for item in data.default]
  data.student = [student[item] for item in data.student]
  x = data[['student','balance','income']] #features del vector de entrada
  y = data['default'] #Categoria a ser predecida
  return x,y

#Funcion para modificar dataset original e introducirle valores nulos
def modifySet(x):
  
  x2 = x.copy()

  # Recorrer todos los registros del training set y con una probabilidad de 0.2
  # Elimina los valores delos features balancee income para una instancia en especifico
  i = 0
  for row in x2:
    if (random.random()) < 0.2:
      np.put(x2[i],[1,2],[np.nan,np.nan])
    i = i + 1

  #Para la columna de features 'balance', recorrer todos los valores y con una proababilidad de 0.1,eliminar el valor correspondiente
  z = 0
  for row in x2:
    if (random.random()) < 0.1:
      np.put(x2[z],1,np.nan)
    z = z + 1
 
  return x2

#Rellena los datos faltantes a traves del uso de los valores promedio de cada feature.
def supplyValuesMean(x):
  x2 = x.copy()
  mean = np.nanmean(x2,axis=0)
  i = 0
  for row in x2:
    if np.isnan(row[1]):
      np.put(x2[i],1,mean[1])
    if np.isnan(row[2]):
      np.put(x2[i],2,mean[2])
    i = i + 1
  return x2

#Rellena los datos faltantes a traves del uso de valores generados a traves de regresion lineal.
def supplyValuesLinearReg(x):
  x3 = x.copy()
  df = pd.DataFrame(data=x3, columns=["student","balance", "income"])
  df['balance'].interpolate(method='linear', inplace=True)
  df['income'].interpolate(method='linear', inplace=True)
  return x3

#Rellena los datos faltantes a traves de la busqueda de los 20 vecinos mas cercanos de cada instancia faltante. 
def supplyValuesKkNeigh(x):
  x4 = x.copy()
  imputer = KNNImputer(n_neighbors=20)
  x4 = imputer.fit_transform(x4)
  return x4

def applySciKitTools(x,y,x_test,y_test):
  #Usando Regresion lineal
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


def main():
  x,y = readFile()
  # Creando vector de entrenamiento
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  x_train = x_train.values  #dataset datos aletorios
  y_train = y_train.values  #dataset datos aletorios
  y_test = y_test.values

  # Llama funcion donde se modificara el set de entrenamiento y se regresara un set nuevo
  x2_train = modifySet(x_train)

  # Llama funcion donde se relleneran datos faltantes de diferente manera
  x2Mod = supplyValuesMean(x2_train)
  x3Mod = supplyValuesLinearReg(x2_train)
  x4Mod = supplyValuesKkNeigh(x2_train)

  # Aplicar las tecnicas de clasificacion de regresion lineal,
  # arboles dedecision y 50-Nearest Neighbors, 
  # empleando las funciones proporcionadas por SciKit-Learn.
  print("T'1:")
  applySciKitTools(x2Mod,y_train,x_test,y_test)
  print("T'2:")
  applySciKitTools(x3Mod,y_train,x_test,y_test)
  print("T'3:")
  applySciKitTools(x4Mod,y_train,x_test,y_test)
  #Original
  print("T Original:")
  applySciKitTools(x_train,y_train,x_test,y_test)


if __name__ == '__main__':
  main()