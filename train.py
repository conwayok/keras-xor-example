import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# xor
input_dataset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_dataset = np.array([[0], [1], [1], [0]])

# Fit the model
model.fit(input_dataset, output_dataset, epochs=2000, batch_size=4)

print(model.predict(input_dataset))
