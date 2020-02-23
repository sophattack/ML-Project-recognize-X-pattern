# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import argparse

import numpy as np
import matplotlib.pyplot as plt
from dispkernel import dispKernel


def load_data(type, delimiter=","):
    data_name = '{}data.csv'.format(type)
    label_name = '{}label.csv'.format(type)
    data = np.loadtxt(data_name, delimiter=delimiter)
    label = np.loadtxt(label_name)

    return data, label


def MSE(W, b, x, y, acttype):
    y_predicted = np.matmul(x, W) + b
    if acttype == 'relu':
        y_predicted = np.maximum(y_predicted, 0)
    total_error = np.mean(np.square(y_predicted - y))
    return total_error


def CE(W, b, x, y):
    N = x.shape[0]
    y_predicted = np.matmul(x, W) + b
    total_error = (np.matmul(np.transpose((1.0 - y)), y_predicted) + np.sum(np.log(1.0 + np.exp(-y_predicted)))) / N
    total_error = np.reshape(total_error, (1,))
    return total_error


def gradMSE(W, b, x, y, acttype):
    N = x.shape[0]
    y_predicted = np.matmul(x, W) + b
    H = np.ones((y.shape))
    w_deri = np.matmul(np.transpose(x), y_predicted - y) / N
    b_deri = (np.matmul(np.matmul(np.transpose(W), np.transpose(x)), H) - np.matmul(np.transpose(H), y) + np.matmul(np.transpose(H), H) * b) / N

    if acttype == "relu":
        total_error = MSE(W, b, x, y, acttype)
        if total_error < 0:
            w_deri = np.zeros((9, 1))
            b_deri = 0.0

    w_deri = np.reshape(w_deri, (9, 1))
    b_deri = np.reshape(b_deri, (1,))

    return w_deri, b_deri


def gradCE(W, b, x, y):
    N = x.shape[0]
    y_predicted = np.matmul(x, W) + b
    H = np.transpose(np.ones(y.shape))
    w_deri = np.matmul(np.transpose(x), (1.0 - y - np.exp(-y_predicted) / (1.0 + np.exp(-y_predicted)))) / N
    b_deri = np.matmul(H, (1.0 - y - np.exp(-y_predicted) / (1.0 + np.exp(-y_predicted)))) / N

    w_deri = np.reshape(w_deri, (9, 1))
    b_deri = np.reshape(b_deri, (1,))

    return w_deri, b_deri


def actfunction(type, x):
    if type == 'sigmoid':
        res = 1/(1 + np.exp(-x))
    elif type == 'relu':
        res = np.maximum(x, 0)
    elif type == 'linear':
        res = x
    else:
        res = -1
    return res


def grad_descent(W, b, trainData, trainLabel, valData, valLabel, lr, numEpoch, acttype):
    w_opt = W
    b_opt = b
    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []

    for i in range(numEpoch):
        if acttype in ['linear', 'relu']:
            w_deri, b_deri = gradMSE(w_opt, b_opt, trainData, trainLabel, acttype)
        elif acttype == 'sigmoid':
            w_deri, b_deri = gradCE(w_opt, b_opt, trainData, trainLabel)
        w_opt -= w_deri * lr
        b_opt -= b_deri * lr
        if acttype in ["linear", 'relu']:
            cost = MSE(w_opt, b_opt, trainData, trainLabel, acttype)
        elif acttype == "sigmoid":
            cost = CE(w_opt, b_opt, trainData, trainLabel)
        trainLoss.append(cost)
        num_val = 0.0
        prediction = actfunction(acttype, np.matmul(trainData, w_opt) + b_opt)

        for j in range(len(trainLabel)):
            if prediction[j] >= 0.5:
                prediction[j] = 1.0
            else:
                prediction[j] = 0.0
            if prediction[j] == trainLabel[j]:
                num_val += 1.0
        accuracy = num_val / len(trainLabel)
        trainAcc.append(accuracy)

        num_val = 0.0
        if acttype in ["linear", 'relu']:
            val_cost = MSE(w_opt, b_opt, valData, valLabel, acttype)
        elif acttype == "sigmoid":
            val_cost = CE(w_opt, b_opt, valData, valLabel)
        valLoss.append(val_cost)
        prediction = actfunction(acttype, np.matmul(valData, w_opt) + b_opt)
        for k in range(len(valLabel)):
            if prediction[k] > 0.5:
                prediction[k] = 1.0
            else:
                prediction[k] = 0.0
            if prediction[k] == valLabel[k]:
                num_val += 1.0
        val_accuracy = num_val / len(valLabel)
        valAcc.append(val_accuracy)

        print("iter: " + str(i) + " cost: " + str(cost) + " training accuracy: " + str(accuracy) + " validation accuracy: " + str(val_accuracy) )

    return w_opt, b_opt, trainLoss, trainAcc, valLoss, valAcc


def plot_graph(title_name, x_label_name, y_label_name, trainData, validData):
    plt.figure()
    plt.title(title_name)
    plt.plot(np.array(np.arange(len(trainData))), trainData, color='orange', label='training')
    plt.plot(np.array(np.arange(len(validData))), validData, color='blue', label='validation')
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.legend()
    plt.show()


def main(args):
    train_data, train_label = load_data(args.trainingfile)
    val_data, val_label = load_data(args.validationfile)
    train_label = np.reshape(train_label, (args.numtrain, 1))
    val_label = np.reshape(val_label, (args.numvalid, 1))
    np.random.seed(args.seed)
    W = np.random.rand(train_data.shape[1], 1)
    b = np.random.rand(1)
    print(W)
    print(b)
    w_opt, b_opt, trainLoss, trainAcc, valLoss, valAcc = grad_descent(W, b, train_data, train_label, val_data, val_label, args.learningrate, args.numepoch, args.actfunction)
    plot_graph("Training and Validation Loss Curve", "Number of Epochs", "Training and Validation Loss", trainLoss, valLoss)
    plot_graph("Training and Validation Accuracy Curve", "Number of Epochs", "Training and Validation Accuracy", trainAcc,
               valAcc)
    dispKernel(w_opt, 3, 99)



if __name__ == '__main__':

    # Command Line Arguments

    parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
    parser.add_argument('--trainingfile', help='name stub for training data and label output in csv format', default="train")
    parser.add_argument('--validationfile', help='name stub for validation data and label output in csv format', default="valid")
    parser.add_argument('--numtrain', help='number of training samples', type=int, default=200)
    parser.add_argument('--numvalid', help='number of validation samples', type=int, default=20)
    parser.add_argument('--seed', help='random seed', type=int, default=396)
    parser.add_argument('--learningrate', help='learning rate', type=float, default=0.06)
    parser.add_argument('--actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'], default='sigmoid')
    parser.add_argument('--numepoch', help='number of epochs', type=int, default=100)

    args = parser.parse_args()

    traindataname = args.trainingfile + "data.csv"
    trainlabelname = args.trainingfile + "label.csv"

    print("training data file name: ", traindataname)
    print("training label file name: ", trainlabelname)

    validdataname = args.validationfile + "data.csv"
    validlabelname = args.validationfile + "label.csv"

    print("validation data file name: ", validdataname)
    print("validation label file name: ", validlabelname)

    print("number of training samples = ", args.numtrain)
    print("number of validation samples = ", args.numvalid)

    print("learning rate = ", args.learningrate)
    print("number of epoch = ", args.numepoch)

    print("activation function is ",args.actfunction)
    

    main(args)