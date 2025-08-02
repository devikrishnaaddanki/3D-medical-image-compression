import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input, Conv2D, UpSampling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from skimage import feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import keras
from PIL import Image


path = "Dataset"
X = []
Y = []

def getOmlsvd(path):
    image = Image.open(path).convert('L')
    image_array = np.array(image)
    #get SVD
    U, S, V = np.linalg.svd(image_array)
    S = np.diag(S)
    k = 50
    #calculate optimial values
    oml = U[:, :k] @ S[0:k,:k] @ V[:k, :]
    oml = Image.fromarray(oml.astype(np.uint8))
    oml.save('oml.jpg')
    oml = cv2.imread('oml.jpg')
    oml = cv2.resize(oml, (128, 128), cv2.INTER_LANCZOS4)
    return oml

'''
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        oml = getOmlsvd(root+"/"+directory[j])
        img = cv2.imread(root+"/"+directory[j])
        img = cv2.resize(img, (128, 128), cv2.INTER_LANCZOS4)
        X.append(oml)
        Y.append(img)
        print(str(j)+" "+str(oml.shape))
            

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)
print(Y.shape)
print(np.unique(Y, return_counts=True))

np.save('model/X',X)
np.save('model/Y',Y)
'''
X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32')
X = X/255
Y = Y.astype('float32')
Y = Y/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
#Y = to_categorical(Y)
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

input_img = Input(shape=(128, 128, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='mean_squared_error')
if os.path.exists("model/decoder_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/decoder_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, batch_size = 64, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/decoder_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/decoder_weights.hdf5")

compress = cv2.imread("Dataset/0001a_frontal.png")
compress_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
cv2.imwrite('jpg_compress.jpg', compress, compress_param)

test = getOmlsvd("Dataset/0001a_frontal.png")
temp = []
temp.append(test)
test = np.asarray(temp)
test = test.astype('float32')
test = test/255
predict = model.predict(test)
predict = predict[0]
cv2.imshow("a", predict)
cv2.imshow("b", predict*255)
cv2.waitKey(0)
