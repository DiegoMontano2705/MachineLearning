import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from sklearn import datasets


#Funcion logistica
def logit(t):
  return 1.0 / (1 + np.exp(-t))

def batchGradientDescent(x,y,alpha,threshold):
  #Añadiendo una columna con 1's a x
  m1,n1 = x.shape
  aux = np.ones(m1)
  aux = aux.reshape(m1,1)
  x = np.append(aux,x,axis=1)
  #Creando beta
  m2,n2 = x.shape
  aux = np.ones(n2)
  beta = np.zeros(n2)
  beta = beta.reshape(n2,1)         #beta = np.ones((np.shape(x)[1],1))
  #beta[0] = 0.5
  #Cambiando forma de y sin perder sus datos
  y = y.reshape(-1,1)
  x_transpose = x.transpose()
  j = 0
  betasResta = 1
  #Algoritmo
  while betasResta > threshold:
    B0 = beta
    t = np.dot(x,beta)
    functLogistica = logit(t) - y
    beta = beta - (alpha/m2)*((np.dot(x_transpose,functLogistica))/m2)
    betasResta = np.linalg.norm(B0-beta)
    j = j + 1
  return beta,j


def tasaDeError(x,y,beta):
  mse = 0
  j = 0
  b = beta.transpose()
  probTotal = []
  vectorProb = []
  #Se obtienen todas las predicciones, se guardan y despues de clasifican dependiendo su valor
  for i in x:
    prob = logit(np.dot(b,i))
    vectorProb.append(prob)
    if prob >= 0.5:
      probTotal.append(1.0)
    else:
      probTotal.append(0)
    #Se obtiene taza de error Sum(I(Y^,y))
  for i in y:
    if probTotal[j] != i :
      mse = mse + 1
    j = j+1
  mse = (1/len(x))* mse
  #MSE = Taza de error , Probtotal = Y^ , VectorProb = p^
  return mse,probTotal,vectorProb

def sciKitLearnMethod(x,y,xTest,yTest,alpha,threshold):
  model = LogisticRegression()
  #Entrenando modelo
  model.fit(x,y)
  y_pred = model.predict(x)
  #Encontrando valores de theta
  beta1 = model.coef_
  beta0 = model.intercept_
  print("Vector beta usando Scikit-Learn: Coeficiente =",beta1,"Intercepcion =",beta0)
  #Encontrando precison del modelo
  accuracy = accuracy_score(y,y_pred)
  print("Taza de error usando Scikit-Learn:",accuracy)

  
def main():
  # genero.txt  Default.txt
  dataSet =  sys.argv[1] # param1: nombre del archivo del dataset.
  trainSetProc = sys.argv[2]#  param2: porcentaje de elementos del training set (ocupar valores enteros).  
  testSetProc =  sys.argv[3]#  param3: porcentaje de elementos del test set (ocupar valores enteros).
  alpha = sys.argv[4] # param4: valor de α.  0.5 o 0.0005
  threshold =  sys.argv[5] # param5: valor del threshold (1x10^-4).0.0001
  semillaGem = sys.argv[6] # param6: semilla del generador de números pseudoaleatorios. 37

  #Links a los archivos que se encuentran en Github y se guarda la información en una matriz
  if dataSet == "Default.txt":
    url = "https://raw.githubusercontent.com/DiegoMontano2705/MachineLearning/main/Parcial1/Default.txt"
    data = pd.read_csv(url, sep='\t')
    default = {'Yes': 0, 'No': 1}
    student = {'Yes': 0, 'No': 1}
    data.default = [default[item] for item in data.default]
    data.student = [student[item] for item in data.student]
    x = data[['student','balance','income']] #features del vector de entrada
    y = data['default'] #Categoria a ser predecida
  elif dataSet == "genero.txt":
    url = "https://raw.githubusercontent.com/DiegoMontano2705/MachineLearning/main/Parcial1/genero.txt"
    data = pd.read_csv(url)
    gender = {'Male': 0, 'Female': 1}
    data.Gender = [gender[item] for item in data.Gender]
    x = data[['Height','Weight']] #features del vector de entrada
    y = data['Gender'] #Categoria a ser predecida
  else:t
    print("Cant process file")

  #Creando vector de entrenamiento
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=testSetProc, random_state=semillaGem)
  x_train = x_train.values  #dataset datos aletorios
  y_train = y_train.values  #dataset datos aletorios
  y_test = y_test.values
  betaFinal,n = batchGradientDescent(x_train,y_train,alpha,threshold)
  #Error cuadratico medio
  m1,n1 = x_test.shape
  aux = np.ones(m1)
  aux = aux.reshape(m1,1)
  x_test = np.append(aux,x_test,axis=1)
  y_test = y_test.reshape(-1,1)
  
  pred,y_gorrito,vectorProb = tasaDeError(x_test,y_test,betaFinal)
  
  #Valores finales
  print("Vector beta =",betaFinal)
  print("Interacciones necesarias para entrenar modelo =",n)
  print("Valor de alpha empleado =",alpha)
  print("Valor de threshold empleado =",threshold)
  print("Taza de error que tiene el modelo =",pred)

  #Vectores finales guardados en archivo
  file = open('vectores.txt', 'w')
  file.write("Vector X\n")
  np.savetxt(file,x_test,fmt='%1.3f')
  file.write("Vector Y_^\n")
  file.write(str(y_gorrito))
  file.write('\n')
  file.write("Vector Y\n")
  np.savetxt(file,y_test, fmt='%1.3f')
  file.write("Vector Probabilidades(P^) Y\n")
  file.write(str(vectorProb))
  file.close()

  #Grafica de la clasificacion para archivo Default.txt
  #if dataSet == "Default.txt":
    #Y^ vs X

  print("\nRegresion Logistica con Scikit-Learn:")
  sciKitLearnMethod(x_train,y_train,x_test,y_test,alpha,threshold)


if __name__ == '__main__':
  main()