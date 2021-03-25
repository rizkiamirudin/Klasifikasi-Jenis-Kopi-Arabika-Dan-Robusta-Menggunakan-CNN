import os 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from keras.models import Sequential, load_model

img_width, img_height = 224, 224 
model_path = '224x224/coba1/models/model(epoch=100, lr=0.001, Op=adam)/model.h5' 
model_weights_path = '224x224/coba1/models/model(epoch=100, lr=0.001, Op=adam)/weights.h5' 
model = load_model(model_path) 
model.load_weights(model_weights_path)

def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: arabika")
    elif answer == 1:
        print("Label: robusta")
        
    return answer

arabika_t = 0
arabika_f = 0
robusta_t = 0 
robusta_f = 0

for i, ret in enumerate(os.walk('224x224/coba1/validation/arabika')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: arabika")
        result = predict(ret[0] + '/' + filename)
        if result == 0:
            arabika_t += 1
        else:
            arabika_f += 1
            
for i, ret in enumerate(os.walk('224x224/coba1/validation/robusta')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: robusta")
        result = predict(ret[0] + '/' + filename)
        if result == 1:
            robusta_t += 1
        else:
            robusta_f += 1
            
print("True Arabika: ", arabika_t)
print("False Arabika: ", arabika_f)
print("True Robusta: ", robusta_t)
print("False Robusta: ", robusta_f)