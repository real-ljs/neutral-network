import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, datasets, optimizers

os.environ["TF__CPP_MIN_LOG_LEVEL"] = "2"


def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x / 255.0
    x_test = x_test / 255.0
    return (x, y), (x_test, y_test)


class Matmul():
    def __init__(self):
        self.men = {}

    def forword(self, x, W):
        h = np.matmul(x, W)
        self.men = {'x': x, 'W': W}
        return h

    def backword(self, grad_y):
        x = self.men['x']
        W = self.men['W']
        grad_x = np.matmul(grad_y, W.T)
        grad_W = np.matmul(x.T, grad_y)
        return grad_x, grad_W


class Relu():
    def __init__(self):
        self.men = {}

    def forword(self, x):
        self.men['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))

    def backword(self, grad_y):
        x = self.men['x']
        return (x > 0).astype(np.float32) * grad_y


class Softmax():
    def __init__(self):
        self.men = {}
        self.epsilen = 1e-12

    def forword(self, x):
        # print(x.shape)
        # print(x)
        x_exp = np.exp(x)
        sum = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (sum + self.epsilen)
        self.men['out'] = out
        self.men['x_exp'] = x_exp
        return out

    def backford(self, grad_y):
        s = self.men['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)
        tmp = np.squeeze(tmp, axis=1)
        softmax_grad = -tmp + grad_y * s
        return softmax_grad


class Cross_entropy():
    def __init__(self):
        self.epsilen = 1e-12
        self.men = {}

    def forword(self, x, labels):
        log_prob = np.log(x + self.epsilen)
        sum = np.sum(-log_prob * labels, axis=1)
        loss = np.mean(sum)
        self.men['x'] = x
        return loss

    def backword(self, labels):
        x = self.men['x']
        return -1 / (x + self.epsilen) * labels


class myModel():
    def __init__(self):
        self.W1 = np.random.normal(size=[28 * 28 + 1, 200])
        # self.W2 = np.random.normal(size=[100, 10])
        self.W3 = np.random.normal(size=[200, 10])
        self.mul_h1 = Matmul()
        self.relu1 = Relu()
        self.mul_h2 = Matmul()
        # self.relu2 = Relu()
        self.mul_h3 = Matmul()
        self.softmax = Softmax()
        self.cross_en = Cross_entropy()

    def forword(self, x, labels):
        x = x.reshape(-1, 28 * 28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        self.h1 = self.mul_h1.forword(x, self.W1)
        self.h1_relu = self.relu1.forword(self.h1)
        # self.h2 = self.mul_h2.forword(self.h1_relu, self.W2)
        # self.h2_relu = self.relu2.forword(self.h2)
        # print(self.W3.shape)
        self.h3 = self.mul_h3.forword(self.h1_relu, self.W3)
        # print(self.h3.shape)
        # print(self.h2.shape)
        self.h3_soft = self.softmax.forword(self.h3)
        self.loss = self.cross_en.forword(self.h3_soft, labels)

    def backword(self, labels):
        self.loss_grad = self.cross_en.backword(labels)
        self.h3_soft_grad = self.softmax.backford(self.loss_grad)
        self.h3_grad, self.W3_grad = self.mul_h3.backword(self.h3_soft_grad)
        # print(self.h3_grad.shape)
        # self.h2_relu_grad = self.relu2.backword(self.h3_grad)
        # print(self.h2_relu_grad.shape)
        # self.h2_grad, self.W2_grad = self.mul_h2.backword(self.h2_relu_grad)
        # print(self.h2_grad.shape)
        self.h1_relu_grad = self.relu1.backword(self.h3_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backword(self.h1_relu_grad)


model = myModel()


def cp_accuracy(prob, labels):
    predic = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predic == truth)


def one_train(model, x, y):
    model.forword(x, y)
    model.backword(y)
    model.W1 -= 2e-5 * model.W1_grad
    # model.W2 -= 1e-5 * model.W2_grad
    model.W3 -= 2e-5 * model.W3_grad
    loss = model.loss
    accuracy = cp_accuracy(model.h3_soft, y)
    return loss, accuracy


def test(model, x, y):
    model.forword(x, y)
    loss = model.loss
    accuracy = cp_accuracy(model.h3_soft, y)
    return loss, accuracy


train_data, test_data = mnist_dataset()
# print(train_data[0].shape)
# print(train_data[1].shape)
train_label = np.zeros(shape=[train_data[0].shape[0], 10])

test_label = np.zeros(shape=[test_data[0].shape[0], 10])
# print(train_data[0])
# print(train_data[1])
# print(test_data[1])
train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1
test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for i in range(500):
    loss, accuracy = one_train(model, train_data[0], train_label)
    train_loss.append(loss.item())
    train_acc.append(accuracy.item())
    # loss, accuracy = test(model, test_data[0], test_label)
    print('第', i, '次', ':', 'loss: ', loss, 'accuracy: ', accuracy)
    # loss, accuracy = test(model, test_data[0], test_label)
    # test_loss.append(loss.item())
    # test_acc.append(accuracy.item())

with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))

with open("./train_acc.txt", 'w') as train_ac:
    train_ac.write(str(train_acc))

with open("./test_loss.txt", 'w') as test_los:
    test_los.write(str(test_loss))

with open("./test_acc.txt", 'w') as test_ac:
    test_ac.write(str(test_acc))

loss, accuracy = test(model, test_data[0], test_label)
print('test loss:', loss, 'accuracy: ', accuracy)
