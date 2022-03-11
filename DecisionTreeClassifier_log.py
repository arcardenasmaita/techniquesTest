import pandas as pd

# first fold 1268
# data = pd.read_csv('simple_data.csv', sep=',',header=None)
# #print(data.head())
#
# act_enc = pd.get_dummies(data[1], prefix='act')
# #print(act_enc.head())
#
# time_y = data[2].copy()
# time_y = time_y.drop([0]).reset_index(drop=True)
#
# data = pd.concat([data[0], act_enc, data[2],time_y ], axis=1, ignore_index=True)
# print(data.head())


#### LASSO regression

import numpy as np  # Herramientas para la manipulación rápida de matrices de estructura
import matplotlib.pyplot as plt  # Dibujo visual
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV  # Regresión de lazo, validación cruzada de LassoCV para lograr la selección alfa, LassoLarsCV basado en validación cruzada de regresión de ángulo mínimo para lograr la selección alfa

# Generar matriz X e Y
# dataMat = np.array(data)
# X = data.loc[:,0:10]   # Variable x
# y = data.loc[:,11]   # Variable y
#
# print(type(X))
# print(type(y))

# # ======== Regresión de lazo ========
# model = Lasso(alpha=0.01)  # Ajuste el alfa para lograr el grado de ajuste
# # model = LassoCV () # LassoCV ajusta automáticamente el alfa para seleccionar el mejor alfa.
# # model = LassoLarsCV () # LassoLarsCV ajusta automáticamente el alfa para seleccionar el mejor alfa
# model.fit(X, y)   # Modelado de regresión lineal
# print('Matriz de coeficientes: \ n',model.coef_)
# print('Modelo de regresión lineal: \ n',model)
# # print ('Best alpha:', model.alpha_) # Solo válido cuando se usa LassoCV, LassoLarsCV
# # Usar predicción del modelo
# predicted = model.predict(X)
#
# # Dibujar un diagrama de dispersión Parámetros: x eje horizontal y eje vertical
# plt.scatter(X, y, marker='x')
# plt.plot(X, predicted,c='r')
#
# # Dibuja las coordenadas del eje x y del eje y
# plt.xlabel("x")
# plt.ylabel("y")
#
# # Mostrar gráficos
# plt.show()


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# decision regressor
data = pd.read_csv('data/simple_data_short.csv', sep=',', header=None)

act_y = data[1].copy()
act_y = act_y.drop([0]).reset_index(drop=True)
act_y = act_y.append(pd.Series(['0']))

X = pd.concat([data[1], data[2], data[3]], keys=['act', 't1', 't2'], axis=1)
y = pd.DataFrame(act_y, columns=["y"])  # Variable y
y = y.astype('str')

print(X.head())
print(y.head())

model = DecisionTreeClassifier()

model.fit(X, y)
y_pred = model.predict(X)
print("Accuracy:", metrics.accuracy_score(y, y_pred))

texto_modelo = export_text(
    decision_tree=model,
    feature_names=['act', 't1', 't2']
)
print(texto_modelo)

# Estructura del árbol creado
# ------------------------------------------------------------------------------

print(f"Profundidad del árbol: {model.get_depth()}")
print(f"Número de nodos terminales: {model.get_n_leaves()}")

fig = plt.figure(figsize=(35, 40))
_ = tree.plot_tree(model,
                   feature_names=["act", "t1", "t2"],
                   #class_names=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                   filled=True)

fig.savefig("tree_class.png")
