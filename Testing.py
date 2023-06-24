import pickle
import time
from tensorflow.keras.callbacks import TensorBoard

Name = f'cat-dog-panda-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir = f'logs\\{Name}\\')


X = pickle.load(open('X.pkl','rb'))
y = pickle.load(open('y.pkl','rb'))

X=X/255
print(X.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(3, activation='sigmoid'))


model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X, y, epochs = 7, validation_split=0.1, batch_size = 32, callbacks=[tensorboard])

model.save('3x3x64-catvsdogvspanda.model')

