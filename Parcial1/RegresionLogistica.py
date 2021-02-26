#Diego Fernando Montaño Pérez A01282874
#Creado: 22/02/2021
import sys
import numpy as np

dataSet = sys.argv[1] # param1: nombre del archivo del dataset.
trainSetProc = sys.argv[2] # param2: porcentaje de elementos del training set (ocupar valores enteros).  
testSetProc = sys.argv[3] # param3: porcentaje de elementos del test set (ocupar valores enteros).
alpha = sys.argv[4] # param4: valor de α.
threshold= sys.argv[5] # param5: valor del threshold (1x10^-4).
semillaGem = sys.argv[6] # param6: semilla del generador de números pseudoaleatorios.

a = np.arange(15).reshape(3, 5)
print(a)
