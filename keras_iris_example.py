import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense

# 1) import data
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# 2) prepare inputs: remove the last column with the iris names
input_x = iris.drop(4,axis=1)

# 3) prepare outputs
# give me all the unique classes -> convert to list -> convert list into a dictionary of values:index
#{'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
key_value = {v: k for k, v in enumerate(list(iris[4].unique()))}
# create an output series with the values replaced by indexes
output_class_int = iris.replace({4:key_value})[4]
# now you can use to_categorical to convert the output categories to arrays
output_y = keras.utils.to_categorical(output_class_int,len(key_value))

# 4a) Create the model
model = keras.models.Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 4b) Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4c) Fit the model
model.fit(input_x, output_y, epochs=150, batch_size=15, verbose=1)
score = model.evaluate(input_x, output_y, batch_size=15)
score
