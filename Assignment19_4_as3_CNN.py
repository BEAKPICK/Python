import pandas as pd
import numpy as np
from keras.models import Sequential #순차모델, 선형으로 레이어 쌓기
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# get data from csv file
data = pd.read_csv('C:/Users/User/PycharmProjects/AIHW4/as3.csv', encoding="utf-8-sig")
data1 = np.array(data)

# split x and y from data
X = list(map(lambda x: x[1:], data1))
Y = list(map(lambda x: x[0], data1))

# train/test/validation random split 작성
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=40)

# the name of class(label)
targetattribute_label = ['horeca','retail'] #str, for printing
label_to_calculate = [0,1] #int, real label value

# change python list to numpy array
X_train = np.asarray(x_train)
X_test = np.asarray(x_test)
X_valid = np.asarray(x_valid)
Y_train = np.asarray(y_train)
Y_test = np.asarray(y_test)
Y_valid = np.asarray(y_valid)

#normalize inputs
X_train = X_train / 112151
X_test = X_test / 112151
Y_train = Y_train -1
Y_test = Y_test -1
Y_valid = Y_valid -1

# change numpy array shape [sample, features]
# to [sample, row, columns(features), channels] 작성
train_data = X_train.reshape(-1, 1, 7, 1)
test_data = X_test.reshape(-1, 1, 7, 1)
validation_data = X_valid.reshape(-1, 1, 7, 1)

# transform Y to one hot vector 작성
train_label = np_utils.to_categorical(Y_train)
test_label = np_utils.to_categorical(Y_test)
validation_label = np_utils.to_categorical(Y_valid)

# total number of class(label) 작성
num_classes = test_label.shape[1]

# 하단에 나머지 작성
def cnn_model():
    model = Sequential()
    model.add(Conv2D(7, (1, 3), input_shape=(1, 7, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = cnn_model()
print(model.summary())
print('\n')

model.fit(train_data, train_label, validation_data=(validation_data, validation_label), epochs=10, batch_size=128, verbose=1)

print('\nEvaluation')
scores = model.evaluate(test_data, test_label, verbose=1)

print('\nPrediction')
y_pred = model.predict(test_data, verbose=1)
Y_pred = np.argmax(y_pred, axis =1)

print('\nCNN on mnist')
prec = precision_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
f1 = f1_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
rec = recall_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
prf = pd.DataFrame([prec, rec, f1], columns=targetattribute_label, index=['Precision', 'Recall', 'F1'])

print(prf.iloc[0], '\n');print(prf.iloc[1], '\n');print(prf.iloc[2],'\n')
print("\nTotal Accuracy: {:.2f}".format(scores[1]*100))