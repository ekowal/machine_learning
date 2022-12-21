from mnist_loader import load_data_wrapper
from neural_network import NeuralNetwork

if __name__=="__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    test_data = list(test_data)
    test = [y for (x, y) in test_data]

    net = NeuralNetwork([784, 40, 40, 10], act_func_names="sigmoid", weights_init="random")
    net.fit(training_data, validation_data, epochs=10, c=0.2, verbose=True)
    preds = net.predict(test_data)
    
    print("Nr of correctly classified samples:", sum(int(y == y_hat) for (y, y_hat) in zip(test, preds)))