# CNN for the cifar10 Dataset
import numpy as np
import pandas as pd
from keras.models import Sequential #순차모델, 선형으로 레이어 쌓기
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


# load data of shape [samples_number, width(row)*height(column)] 2D matrix
x_train = pd.read_csv('C:/Users/User/PycharmProjects/AIHW4/x_train.csv', encoding="utf-8-sig")
y_train = pd.read_csv('C:/Users/User/PycharmProjects/AIHW4/y_train.csv', encoding="utf-8-sig")
x_test = pd.read_csv('C:/Users/User/PycharmProjects/AIHW4/x_test.csv', encoding="utf-8-sig")
y_test = pd.read_csv('C:/Users/User/PycharmProjects/AIHW4/y_test.csv', encoding="utf-8-sig")

train_dataX = np.array(x_train)
X_train = np.asarray(list(map(lambda x: x[0:], train_dataX)))
train_dataY = np.array(y_train)
Y_train = np.asarray(list(map(lambda x: x[0], train_dataY)))

test_dataX = np.array(x_test)
X_test = np.asarray(list(map(lambda x: x[0:], test_dataX)))
test_dataY = np.array(y_test)
Y_test = np.asarray(list(map(lambda x: x[0], test_dataY)))

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# reshape data [samples, width(row)*height(column)*channels]
# to [samples, width(row), height(column), channels] 작성
X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))

# transform Y to one hot vector
Y_train_onehot = np_utils.to_categorical(Y_train)
Y_test_onehot = np_utils.to_categorical(Y_test)

# total number of class(label)
num_classes = Y_train_onehot.shape[1]

# the name of class(label)
mnist_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# slice data 작성
train_data = X_train[:45000] #45000/49999
train_label = Y_train_onehot[:45000]

validation_data = X_train[49000:] #1000/49999
validation_label = Y_train_onehot[49000:]

test_data = X_test[:10000]   #10000/10000
test_label = Y_test_onehot[:10000]


# the name of class(label)
cifar10_label = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck'] #str, for printing
label_to_calculate = [0,1,2,3,4,5,6,7,8,9] #int, real label value

# 하단에 나머지 작성
def cnn_model():
    #define model
    model = Sequential()
    model.add(Conv2D(16,(5,5), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) #overfitting 방지
    model.add(Flatten())    #1차원화 시키기(펼치기), 전결합층에 넣기 전에 필요한 작업
    model.add(Dense(32, activation='relu')) #Dense로 전결합층의 심층 신경망 생성
    model.add(Dense(num_classes, activation='softmax')) #Softmax로 클래스 분류
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#build a model
model = cnn_model()
print(model.summary())
print('\n')

#Fit the model (train 5000 data, validate 1000 data)
model.fit(train_data, train_label, validation_data=(validation_data, validation_label), epochs=1, batch_size=128, verbose=1)

#Evaluation of the model (on 1000 set of test data)
print('\nEvaluation')
scores = model.evaluate(test_data, test_label, verbose=1) #verbose: 얼마나 자세히 정보를 표시할지

#Prediction of the model (on 1000 set of test data)
print('\nPrediction')
y_pred = model.predict(test_data, verbose=1)
Y_pred = np.argmax(y_pred, axis=1)

#Calculate Scores (on 1000 set of test data)
print('\nCNN on mnist')
prec = precision_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
f1 = f1_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
rec = recall_score(np.argmax(test_label, axis=1), Y_pred, labels=label_to_calculate, average=None)
prf = pd.DataFrame([prec, rec, f1], columns=mnist_label, index=['Precision', 'Recall', 'F1'])

print(prf.iloc[0], '\n');print(prf.iloc[1], '\n');print(prf.iloc[2],'\n')
print("\nTotal Accuracy: {:.2f}".format(scores[1]*100))

# Random test & visualization
import matplotlib.pyplot as plt
import random
plt.figure(figsize=(2, 2)) # board size to print
i = random.choice(range(len(Y_pred)))
plt.imshow(test_data[i], cmap="gray")   #X_test: (49999, 32, 32, 3)
plt.title('Prediction: '+cifar10_label[Y_pred[i]]+'  /  True: '+cifar10_label[Y_test[i]])
plt.axis("off")

plt.show()