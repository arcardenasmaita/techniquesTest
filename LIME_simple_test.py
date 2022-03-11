



X = np.random.uniform(0,1, (50, 10, 12))
y = np.random.randint(0,1, (50, 9))

model = Sequential()
model.add(LSTM(12, input_shape=(10, 12)))
model.add(Dense(9, activation='softmax'))
model.compile('adam', 'categorical_crossentropy')
history = model.fit(X, y, epochs=3)

from lime import lime_tabular

explainer = lime_tabular.RecurrentTabularExplainer(
    X, training_labels = y,
    feature_names = ['1','2','3','4','5','6','7','8','9','10','11','12'],
    discretize_continuous = True,
    class_names = ['a','b','c','d','e','f','g','h','i'],
    discretizer = 'decile')

exp = explainer.explain_instance(
    data_row = X[0].reshape(1,10,12),
    classifier_fn = model.predict)

exp.show_in_notebook()