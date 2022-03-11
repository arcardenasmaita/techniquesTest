### http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/nn_multicapa/nn_multicapa.html
### Red Neuronal Multicapa en Keras
### El objetivo es clasificar las imágenes de dígitos (28x28 pixeles) de la popular base de datos MNIST.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

print('backend :', keras.backend.backend())
print('keras version :', keras.__version__)

# Si queremos saber si usaremos el CPU o un el GPU como dispositivo de cómputo, necesitamos comprobarlo a través de tensorflow
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

### Cargando Datos MNIST mediante Keras

# cargar la interfaz a la base de datos que vienen con keras
from tensorflow.keras.datasets import mnist

# lectura de los datos
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

A = train_images[0]  # la primera imagen

# La BD MNIST consta de 60 mil datos datos de entrenamiento y 60 mil de prueba, con sus respectivas etiquetas.
print('Dimensiones del conjunto de entrenamiento: ', train_images.shape)
print('Dimensiones del conjunto de evaluación: ', train_images.shape)
num_data, nrows, ncols = train_images.shape

# La datos para cada clase estan aproximadamentre balanceados: cerca de 6 mil para imágenes para cada clase.
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.hist(train_labels[:], bins=10)
plt.title('Conteo de las etiquetas del conjunto de entrenamiento')
plt.subplot(212)
plt.hist(test_labels[:], bins=10)
plt.title('Conteo de las etiquetas del conjunto de prueba')
plt.show()

# ejemplos de las imágenes para cada clase
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    idx = list(train_labels).index(i)
    plt.imshow(train_images[idx], 'gray')
    plt.title(train_labels[idx])
    plt.axis('off')

plt.show()

# desplegamos las primeras 400’s ocurrencias de las imágenes de los dígitos 1 y 7.

nrowsIm = 20
ncolsIm = 20
numIm = nrowsIm * ncolsIm

digit = 8  # cambiar para ver otro dígito
Indexes = np.where(train_labels == digit)[0][:numIm]

plt.figure(figsize=(12, 12))
for i, idx in enumerate(Indexes[:numIm]):
    plt.subplot(nrowsIm, ncolsIm, i + 1)
    plt.imshow(train_images[idx], 'gray')
    plt.axis('off')

plt.show()

### Preprocesamiento de los Datos
# Usaremos una red clasificadora que usa vectores de entrada unidimensionales (tensores de orden 1). Por lo que preprocesamos cada imagen para:
# transformarla de un tensor de orden 3 de 28 \times 28 \times 128×28×1 (pixeles por renglón, pixeles por columna, número de canales) a un tensor unimensional de 784784 entradas.
# normalizar en valores de cada entrada al intervalo [0, 255][0,255].

train_images = train_images.reshape((60000, -1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, -1))
test_images = test_images.astype('float32') / 255

numIm, szIm = train_images.shape

# Además, las etiquetas, originalmente codificadas en un entero en el conjunto \{0,1,2,\ldots,9\}{0,1,2,…,9}, las transformaremos a un vector de la base canónica e \in \{ e_i\}_{i=1,2,\ldots,10}e
# En al argot de redes neuronales, a esta codificación se denomina one-hot, vectores indicadores (generalmente), o variables categóricas.
# Con keras este mapeo se realiza mediante el siguiente código.

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

### Arquitectura de la red de Percetrones Multicapa

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Las capas Dense son la capas de cálculo de que conectan cada neurona en una capa con todas las salidas de la capa anterior.
model = Sequential()  # Modelo secuencial

# Definir la función de activación como None nos permite introducir un procesamiento (otra capa + layer) antes de invocar a la activación específica. Para cargar una capa de activación podemos usar
import tensorflow.nn as nn
from tensorflow.keras.layers import Activation

# y en su momento usar, por ejemplo'
Activation('relu')  # o Activation(nn.relu)

# añadir al modelo nn la primera capa oculta
model.add(Dense(units=512,  # número de neuronas en la capa
                activation='relu',  # función de activacion: lineal-rectificada
                input_shape=(szIm,)))  # forma de la entrada: (szIm, ) la otra
# dimensión es el tamaño de lote (szBatch),
# que se define en 'fit'
# añadir capa de salida
model.add(Dense(units=10,
                activation='softmax'))  # función de activación: softmax

# El método compile no modifica en nada los párametros de la red. En una red preenetrenada no altera los pesos. De igual forma si se
# compila-entrena un modelo por un número dado de iteraciones (veremos el concepto de época al ver el método para entrenar fit),
# y luego modificamos los parámetros de entrenamiento con compile y volvemos a entrenar, el segundo entrenamiento retomará el
# proceso donde el primer entrenamiento se quedo, cambiando sólo los parámetros del entrenamiento
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

### Selección del algortimo de optimización
# Podemos seleccionar el algoritmo de optimización específico y ajustar sus parámetros. Por ejemplo, si en vez de usar rmsprop en el
# caso anterio, usamos Descenso de Gradente Estocástico (SGD) tipo Nesteron, podemos usar la variante con momentum del tipo Nesterov
# y con decaimiento del factor de aprendizaje después de cada actualización de lote (batch); podemos usar:
from keras import optimizers

# parametros de metodo de optimizacion
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# parametros del procedimiento de aprendizaje (incluye que optimizador usar)
# model.compile(loss='mean_squared_error', optimizer=sgd)

### Entrenamiento
import time

tic = time.time()
history = model.fit(x=train_images,
                    y=train_labels,
                    epochs=1, # original 20
                    shuffle=True,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=2)

print('Tiempo: {} secs'.format(time.time() - tic))
results = model.evaluate(test_images, test_labels)
print(results)

# Para analizar como se comporta los valores de la función objetivo y de las métrica, analizamos el objeto History devuelto por el proceso de entrenamiento.
history_dict = history.history
dictkeys = list(history_dict.keys())

# Estas son la llaves (keys) de los arreglos con los valores monitoreados.
print(dictkeys)

# Veamos el comportamiento del valor de la función objetivo.
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(12, 5))
plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
plt.title('Valor de la función objetivo (loss) en conjuntos de en entrenamiento y validación')
plt.xlabel('Epocas')
plt.ylabel('')
plt.hlines(y=.078, xmin=0, xmax=20, colors='k', linestyles='dashed')
plt.legend()
plt.show()

# y el comportamiento del valor de la métrica monitoreada
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.figure(figsize=(12, 5))
plt.plot(epochs, acc_values, 'b', label='Accuracy')
plt.plot(epochs, val_acc_values, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.hlines(y=.982, xmin=0, xmax=20, colors='k', linestyles='dashed')
plt.legend()
plt.show()

### Visualización del desempeño
y_pred = model.predict(test_images).squeeze()

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import numpy as np
# Import the modules from sklearn.metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

y_test_label = np.argmax(test_labels, 1)
y_pred_label = np.argmax(y_pred, 1)

### Matriz de Confusión
# Los renglones corresponden a las etiquetas reales y las columnas a las predichas.

C = confusion_matrix(y_pred_label, y_test_label)
print(C)

# La matriz de confusion C se puede mostrar como imágen, codificando en color las coocurrencias. Como la lgran mayoría de los digitos son correctamente clasificados, solo veriamos una diagonal dominante y poca diferencia fuera de la misma. Por ello, mejor desplegamos el logaritmo de C+1 (el uno para evitar la indefinición en el cero del logaritmo).

import seaborn as sns

# En escala logaritmica !
plt.figure(figsize=(8, 6.5))
plt.title('matriz de Confusion (escala log)')
sns.heatmap(np.log(C + 1),
            xticklabels=np.arange(10),
            yticklabels=np.arange(10),
            square=True,
            linewidth=0.5, )

plt.show()

### Métricas de Desempeño

# Precisión es la probabilidad de que un dato selecionado aleatoriamente sea relevante
precision_score(y_pred_label, y_test_label, average='macro')

# Recall es la probabilidad de que un dato relevante sea selecionado aleatoriamente.
recall_score(y_pred_label, y_test_label, average='macro')

# F1-score, Media armónica de la precisión y el recall. Penaliza el desbalance entre las métricas P y R
f1_score(y_pred_label, y_test_label, average='macro')

# Coeficiente kappa de Cohen
cohen_kappa_score(y_pred_label, y_test_label)
