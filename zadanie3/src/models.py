from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Convolution2D, Dense, Dropout,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam


class SimpleCNN():
    def __init__(self, input_shape, num_classes):
        model = Sequential()
        model.add(Convolution2D(10, kernel_size=3, activation="relu", input_shape=input_shape))
        model.add(Convolution2D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model


class VGG():
    def __init__(self, input_shape, num_classes, vgg_type=16):
        self.model = Sequential()
        self.model.add(Convolution2D(64, kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape))
        self.model.add(Convolution2D(64, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        
        self.model.add(Convolution2D(128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        
        if vgg_type == 16:
            self.__tri_conv(256)
            self.__tri_conv(512)
            self.__tri_conv(512)
        else:
            self.__qua_conv(256)
            self.__qua_conv(512)
            self.__qua_conv(512)
        
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation="relu"))
        self.model.add(Dense(4096, activation="relu"))
        self.model.add(Dense(num_classes, activation="softmax"))

        self.model.compile(optimizer= Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
 

    def __tri_conv(self, filters):
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D((2, 2), padding="same"))
        
     
    def __qua_conv(self, filters):
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Convolution2D(filters, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D((2, 2), padding="same"))


class AlexNet():
    def __init__(self, input_shape, num_classes):
        model = Sequential()

        model.add(Convolution2D(96, kernel_size = (11,11), strides = 4,
                        padding = 'valid', activation = 'relu',
                        input_shape = input_shape,
                        kernel_initializer= 'he_normal'))

        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        model.add(Convolution2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
                        
        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'same', data_format= None)) 

        model.add(Convolution2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        model.add(Convolution2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        model.add(Convolution2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'same', data_format= None))

        model.add(Flatten())
        model.add(Dense(4096, activation= 'relu'))
        model.add(Dense(4096, activation= 'relu'))
        model.add(Dense(1000, activation= 'relu'))
        model.add(Dense(num_classes, activation= 'softmax'))

        model.compile(optimizer= Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        self.model = model


class SqueezeNet():
    def __init__(self, input_shape, num_classes):

        InputImages = Input(shape=input_shape)

        conv1 = Convolution2D(96, (7, 7), activation='relu')(InputImages)
        maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

        fire2 = self.__fire(maxpool1, squeeze=16, expand=64)
        fire3 = self.__fire(fire2, squeeze=16, expand=64)
        fire4 = self.__fire(fire3, squeeze=32, expand=128)

        maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire4)

        fire5 = self.__fire(maxpool4, squeeze=32, expand=128)
        fire6 = self.__fire(fire5, squeeze=32, expand=128)
        fire7 = self.__fire(fire6, squeeze=48, expand=192)
        fire8 = self.__fire(fire7, squeeze=64, expand=256)

        maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire8)

        fire9 = self.__fire(maxpool8, squeeze=64, expand=256)
        fire9_dropout = Dropout(0.5)(fire9)

        conv10 = Convolution2D(10, (1,1), activation='relu', padding='same')(fire9_dropout)
        avg_pooling10 = AveragePooling2D(padding='same')(conv10)
        flatten = Flatten()(avg_pooling10)
        dense = Dense(num_classes, activation="softmax")(flatten)

        squeezeNet_model = Model(inputs=InputImages, outputs=dense)
        squeezeNet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = squeezeNet_model
        
        
    
    @staticmethod
    def __fire(X, squeeze=16, expand=64):
        fire_squeeze = Convolution2D(squeeze, (1,1), activation='relu', padding='same')(X)
        fire_expand1 = Convolution2D(expand, (1,1), activation='relu', padding='same')(fire_squeeze)
        fire_expand2 = Convolution2D(expand, (3,3), activation='relu', padding='same')(fire_squeeze)
        merge = Add()([fire_expand1, fire_expand2])
        return merge


class ResNet():
    def __init__(self, input_shape, num_classes):
        
        InputImages = Input(shape=input_shape)
        zeroPad1 = ZeroPadding2D((1,1), data_format="channels_first")
        zeroPad1_2 = ZeroPadding2D((1,1), data_format="channels_first")
        
        layer1 = Convolution2D(6, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', data_format="channels_first")
        layer1_2 = Convolution2D(16, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', data_format="channels_first")
        zeroPad2 = ZeroPadding2D((1,1), data_format="channels_first")
        zeroPad2_2 = ZeroPadding2D((1,1), data_format="channels_first")
        
        layer2 = Convolution2D(6, (3, 3), strides=(1,1), kernel_initializer='he_uniform', data_format="channels_first")
        layer2_2 = Convolution2D(16, (3, 3), strides=(1,1), kernel_initializer='he_uniform', data_format="channels_first")
        zeroPad3 = ZeroPadding2D((1,1), data_format="channels_first")
        zeroPad3_2 = ZeroPadding2D((1,1), data_format="channels_first")
        
        layer3 = Convolution2D(6, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', data_format="channels_first")
        layer3_2 = Convolution2D(16, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', data_format="channels_first")
        layer4 = Dense(64, activation='relu', kernel_initializer='he_uniform')
        layer5 = Dense(16, activation='relu', kernel_initializer='he_uniform')
        final = Dense(num_classes, activation='softmax', kernel_initializer='he_uniform')
        
        first = zeroPad1(InputImages)
        second = layer1(first)
        second = BatchNormalization(axis=1)(second)
        second = Activation('relu')(second)

        third = zeroPad2(second)
        third = layer2(third)
        third = BatchNormalization(axis=1)(third)
        third = Activation('relu')(third)

        third = zeroPad3(third)
        third = layer3(third)
        third = BatchNormalization(axis=1)(third)
        third = Activation('relu')(third)
        
        res = Add()([third, second])


        first2 = zeroPad1_2(res)
        second2 = layer1_2(first2)
        second2 = BatchNormalization(axis=1)(second2)
        second2 = Activation('relu')(second2)


        third2 = zeroPad2_2(second2)
        third2 = layer2_2(third2)
        third2 = BatchNormalization(axis=1)(third2)
        third2 = Activation('relu')(third2)

        third2 = zeroPad3_2(third2)
        third2 = layer3_2(third2)
        third2 = BatchNormalization(axis=1)(third2)
        third2 = Activation('relu')(third2)
        
        res2 = Add()([third2, second2])

        res2 = Flatten()(res2)

        res2 = layer4(res2)
        res2 = Dropout(0.4)(res2)
        res2 = layer5(res2)
        res2 = Dropout(0.4)(res2)
        res2 = final(res2)
        model = Model(inputs=InputImages, outputs=res2)

        sgd = SGD(decay=0., learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        self.model = model