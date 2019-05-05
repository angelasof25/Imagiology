import os
from math import ceil

import matplotlib.pyplot as plt
import matplotlib as matplotlib

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np

from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
K.set_image_dim_ordering('tf')


path = f'/home/angela/Transferências/RX_torax2/train/NORMAL'
path_ = f'/home/angela/Transferências/RX_torax2/train/VIRUS'
path__ = f'/home/angela/Transferências/RX_torax2/train/BACTERIA'
PATH = f'/home/angela/Transferências/RX_torax2/'
teste = f'/home/angela/Transferências/RX_torax2//test'


# contar o número de imagens de cada tipo
def directories(path1=path, path2=path_, path3=path__):

    list_dir = os.listdir(path1)
    list_dir2 = os.listdir(path2)
    list_dir3 = os.listdir(path3)
    list_teste1 = os.listdir(teste+'/NORMAL')
    list_teste2 = os.listdir(teste+'/VIRUS')
    list_teste3 = os.listdir(teste+'/BACTERIA')
    normal = 0
    bacteria = 0
    virus = 0
    num_casos_validacao = 0

    for file in list_dir:
        if file.startswith('NORMAL_'):
            normal += 1
    for file in list_dir2:
        if file.startswith('VIRUS_'):
            virus += 1
    for file in list_dir3:
        if file.startswith('BACTERIA_'):
            bacteria += 1

    for file in list_teste1:
        num_casos_validacao += 1
    for file in list_teste2:
        num_casos_validacao += 1
    for file in list_teste3:
        num_casos_validacao += 1

    return normal, virus, bacteria, num_casos_validacao


normal, virus, bacteria, num_casos_validacao = directories()
l = ['normal', 'virus', 'bacteria']
w = [normal, virus, bacteria]
#plt.bar(l, w)
#plt.show()
#print(num_casos_validacao)
num_samples = normal + virus + bacteria

# data generator
def data_generator():

    datagenerator_treino = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                              width_shift_range=0.05, height_shift_range=0.05, shear_range=0.1)

    datagenerator_teste = ImageDataGenerator(rescale=1./255)

    batch_size = 50
    casos_treino = datagenerator_treino.flow_from_directory(PATH+'/train', target_size=(80, 80),
                                                            batch_size=batch_size,
                                                            class_mode='categorical', color_mode='grayscale')
    casos_teste = datagenerator_teste.flow_from_directory(PATH+'/test', target_size=(80, 80),
                                                          batch_size=batch_size,
                                                          class_mode='categorical', color_mode='grayscale', shuffle=False)
    casos_teste.reset()

    return casos_treino, casos_teste


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 1), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def treino_teste(casos_treino, casos_teste):
    batch_size = 50
    model = create_model()
    model.summary()

    checkpointer = ModelCheckpoint(filepath="melhor_modelo2.h5", monitor='val_acc', verbose=1, save_best_only=True)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(casos_treino.classes), casos_treino.classes)
    print(class_weights)

    history = model.fit_generator(casos_treino, steps_per_epoch=ceil(num_samples / batch_size),
                                  epochs=35, validation_data=casos_teste,
                                  validation_steps=ceil(num_casos_validacao / batch_size), callbacks=[checkpointer]
                                  , class_weight=class_weights, shuffle=True, workers=5)



    print_history_accuracy(history)
    print_history_loss(history)

    # Avaliação final com os casos de teste
    scores = model.evaluate_generator(casos_teste, num_casos_validacao / batch_size, verbose=1)

    print('Scores: ', scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("Erro modelo : %.2f%%" % (100 - scores[1] * 100))


def predictions(casos_teste):
    model = load_model('melhor_modelo2.h5')

    pred = model.predict_generator(casos_teste, steps=len(casos_teste), verbose=1)
    pred = np.argmax(pred, axis=1)
    matriz = confusion_matrix(casos_teste.classes, pred)

    plt.figure()
    plot_confusion_matrix(matriz, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(3), ['Normal', 'Virus', 'Bacteria'], fontsize=16)
    plt.yticks(range(3), ['Normal', 'Virus', 'Bacteria'], fontsize=16)
    plt.show()

    acc = np.mean(pred == casos_teste.classes)
    print("Accuracy: %.2f%%" % (acc * 100))

    return model


if __name__ == '__main__':

    casos_treino, casos_teste = data_generator()
   # treino_teste(casos_treino, casos_teste)
    predictions(casos_teste)
