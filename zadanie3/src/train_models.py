import pathlib

from models import *
from make_dataset import prepare_mnist

EPOCHS = 3
CWD = pathlib.Path(__file__).resolve()
PARENT_PATH = CWD.parent.parent

(X_train, y_train), (X_test, y_test) = prepare_mnist()
input_shape, num_classes = X_train[0].shape, y_train[0].shape[0]

models = [
    SimpleCNN(input_shape, num_classes),
    VGG(input_shape, num_classes, vgg_type=16),
    VGG(input_shape, num_classes, vgg_type=19),
    AlexNet(input_shape, num_classes),
    SqueezeNet(input_shape, num_classes),
    ResNet(input_shape, num_classes)
    ]


def train_and_save():
    for model in models:
        name = model.__class__.__name__
        model_compiled = model.model
        model_compiled.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS)
        
        if name == "VGG":
            name = name + "_" + model.vgg_type
        model_compiled.save(PARENT_PATH / "models" / (name + ".h5"))


if __name__=="__main__":
    train_and_save()
        

