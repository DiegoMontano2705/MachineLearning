Aprendizaje autom�tico
Pr�ctica 01
Nota para profesor: Al hacer mi predicci�n y taza de error me estaba dando valores muy peque�os y por mas que buscaba en que me fallaba no encontr� el error. Tambi�n me falto agregar graficas para el archivo Default.txt. A pesar de todo esto entiende bien como funciona el modelo y no me quede con dudas, fueron m�s problemas de programaci�n que tuve pues no hab�a usado mucho Python y me costo algo de trabajo esa parte.
Como correr programa desde consola:
python practica1 genero.txt 0.8 0.2 0.005 0.0001 37 
Resultados en consola que muestra el programa con archivo �genero.txt�: 
Programa
Parte 1: 














En la primera parte del c�digo es donde se leen los datos que manda el usuario al correr el programa y dependiendo del nombre del archivo guarda las tablas de la base de datos de diferente manera pues est�n organizados de diferente manera, tambi�n se modifican valores dentros de las tablas para maneras todo con n�meros, como en el caso de genero.txt donde en la parte de genero de la tabla, se sustituye la palabra �Male� a 0 y Female a �1�, lo mismo pasa en el caso de Default.txt donde la palabra Yes = �0� y No = �1�.
Una ves que se tienen los datos y se separa en las variables X y Y los datos y el objetivo de predicci�n, se crean los vectores de entrenamiento y pruebas en donde de manera aleatorio se escogen quedando un 80% en entrenamiento y un 20% en pruebas. Tambi�n se modifican las tablas agregando una columna de �0� para poder hacer las operaciones m�s adelante.
Parte 2:

Este es el algoritmo de Batch Gradient Descent que programe para obtener el valor de theta, que en este caso lo llame beta. Al inicio les doy forma a las matrices para poder hacer las operaciones correspondientes apoy�ndome de la librer�a numpy para el manejo de matrices. Al final del algoritmo se regresan el vector theta y el numero de iterraci�nes que se requirieron para obtener el un resultado menos al threshold.
Parte 3:

Esta funci�n de tasaDeError me ayuda a encontrar todas las predicciones posibles para despu�s apoyarme con estas mismas para saber la tasa de error del modelo para saber la precisi�n de nuestro modelo. Al final se regresa E, Y^, y p^. 
Parte 4:

La pen�ltima parte del c�digo se hace de nuevo la regresi�n log�stica, pero esta apoy�ndonos de la librer�a Scikit-learn, donde a trav�s de m�todos se crea un modelo y entrena un modelo de manera m�s sencilla, despu�s encontramos los valores de theta que en este caso son el coeficiente e intercepci�n. Por �ltimo, se regresa la precisi�n del modelo

Parte 5:

Por �ltimo, se imprimen los resultados como se encuentra en la parte del inicio del documento y se guardan los vectores correspondientes en un archivo de texto llamado �vectores.txt�
Diego Fernando Monta�o P�rez A01282875


