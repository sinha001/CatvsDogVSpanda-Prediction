import cv2
import keras
import matplotlib.pyplot as plt

import numpy as np
Categories = ['cat', 'dog', 'panda']

def img(path):
    img_arr = cv2.imread(path)
    plt.imshow(img_arr)
    plt.show()
    new_img_arr = cv2.resize(img_arr,(100,100))
    new_img_arr = np.array(new_img_arr)
    new_img_arr = new_img_arr.reshape(1,100,100,3)
    return new_img_arr

model = keras.models.load_model('3x3x64-catvsdogvspanda.model')

prediction = model.predict([img('test_of_model/panda.jpg')])
print(Categories[prediction.argmax()])