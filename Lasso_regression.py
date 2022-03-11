
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Regresión de lazo, validación cruzada de LassoCV para lograr la selección alfa, LassoLarsCV basado en validación cruzada de regresión de ángulo mínimo para lograr la selección alfa

import pandas as pd

#Conjunto de datos de muestra, la primera columna es x, la segunda columna es y, y el modelo de regresión se establece entre x e y
data=[
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
]
print(data)

#Generar matriz X e Y
dataMat = np.array(data)
X = dataMat[:,0:1]   # Variable x
y = dataMat[:,1]   # Variable y



# ======== Regresión de lazo ========
model = Lasso(alpha=0.01)  # Ajuste el alfa para lograr el grado de ajuste
# model = LassoCV () # LassoCV ajusta automáticamente el alfa para seleccionar el mejor alfa.
# model = LassoLarsCV () # LassoLarsCV ajusta automáticamente el alfa para seleccionar el mejor alfa
model.fit(X, y)   # Modelado de regresión lineal
print('Matriz de coeficientes: \ n',model.coef_)
print('Modelo de regresión lineal: \ n',model)
# print ('Best alpha:', model.alpha_) # Solo válido cuando se usa LassoCV, LassoLarsCV
# Usar predicción del modelo
predicted = model.predict(X)

# Dibujar un diagrama de dispersión Parámetros: x eje horizontal y eje vertical
plt.scatter(X, y, marker='x')
plt.plot(X, predicted,c='r')

# Dibuja las coordenadas del eje x y del eje y
plt.xlabel("x")
plt.ylabel("y")

# Mostrar gráficos
plt.show()


