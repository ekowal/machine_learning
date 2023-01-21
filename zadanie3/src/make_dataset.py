from keras.datasets import mnist
from keras.utils import to_categorical

def prepare_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #reshape data to fit model
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return ((X_train, y_train), (X_test, y_test))


def prepare_cifar():
    pass