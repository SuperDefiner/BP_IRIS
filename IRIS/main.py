import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pandas.plotting import radviz

def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2)

    w1 = np.random.randn(n_h, n_x) * np.sqrt(2 / n_x)
    w2 = np.random.randn(n_y, n_h) * np.sqrt(2 / n_h)

    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters



def forward_propagation(X, parameters):

    w1, b1, w2, b2 = parameters['w1'], parameters['b1'], parameters['w2'], parameters['b2']

    z1 = np.dot(w1, X) + b1

    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2

    a2 = 1 / (1 + np.exp(-z2))

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache



def compute_cost(a2, Y, parameters, lambd=0.3):


    m = Y.shape[1]

    log_probs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cross_entropy_cost = - np.sum(log_probs) / m

    w1, w2 = parameters['w1'], parameters['w2']

    l2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    cost = cross_entropy_cost + l2_regularization_cost

    return cost



def backward_propagation(parameters, cache, X, Y, lambd=0.3):


    m = Y.shape[1]

    w1, w2, a1, a2 = parameters['w1'], parameters['w2'], cache['a1'], cache['a2']
    dz2 = a2 - Y
    dw2 = np.dot(dz2, a1.T) / m + (lambd / m) * w2
    db2 = np.mean(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = np.dot(dz1, X.T) / m + (lambd / m) * w1
    db1 = np.mean(dz1, axis=1, keepdims=True)
    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


def update_parameters_with_adam(parameters, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    w1, b1, w2, b2 = parameters.values()
    dw1, db1, dw2, db2 = grads.values()

    vdW1, vdW2 = np.zeros_like(w1), np.zeros_like(w2)
    sdW1, sdW2 = np.zeros_like(w1), np.zeros_like(w2)
    vdb1, vdb2 = np.zeros_like(b1), np.zeros_like(b2)
    sdb1, sdb2 = np.zeros_like(b1), np.zeros_like(b2)

    vdW1 = beta1 * vdW1 + (1 - beta1) * dw1
    vdb1 = beta1 * vdb1 + (1 - beta1) * db1
    vdW2 = beta1 * vdW2 + (1 - beta1) * dw2
    vdb2 = beta1 * vdb2 + (1 - beta1) * db2
    sdW1 = beta2 * sdW1 + (1 - beta2) * np.square(dw1)
    sdb1 = beta2 * sdb1 + (1 - beta2) * np.square(db1)
    sdW2 = beta2 * sdW2 + (1 - beta2) * np.square(dw2)
    sdb2 = beta2 * sdb2 + (1 - beta2) * np.square(db2)

    w1 -= (learning_rate * vdW1) / (np.sqrt(sdW1) + epsilon)
    b1 -= (learning_rate * vdb1) / (np.sqrt(sdb1) + epsilon)
    w2 -= (learning_rate * vdW2) / (np.sqrt(sdW2) + epsilon)
    b2 -= (learning_rate * vdb2) / (np.sqrt(sdb2) + epsilon)

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters



def predict(parameters, x_test, y_test):

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    output = np.where(a2 > 0.5, 1, 0)
    print('预测结果：')
    print(output)
    print("\n")
    print('真实结果：')
    print(y_test)

    accuracy = np.mean(np.all(output == y_test, axis=0)) * 100
    print('准确率：%.2f%%' % accuracy)
    return output



def nn_model(X, Y, n_h, n_input, n_output, num_iterations=10000, print_cost=False):

    np.random.seed(3)

    n_x = n_input
    n_y = n_output

    parameters = initialize_parameters(n_x, n_h, n_y)

    cost_history = []

    for i in range(1, num_iterations + 1):

        a2, cache = forward_propagation(X, parameters)

        cost = compute_cost(a2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters_with_adam(parameters, grads)

        if i % 100 == 0:
            cost_history.append(cost)

        if print_cost and i % 1000 == 0:
            print('迭代第%i次     代价函数：%f' % (i, cost))
            print("-----------------------------------------------")
    return parameters, print_cost, cost_history



def plot_cost_history(cost_history):

    plt.figure('代价函数')
    plt.plot(cost_history)
    plt.title('Cost Function')
    plt.xlabel('Iterations (per 100)')
    plt.ylabel('Cost')
    plt.show()


def result_visualization(x_test, y_test, result):
    """
    :param x_test:测试集特征矩阵，是一个numpy数组。
    :param y_test:测试集标签独热编码矩阵，是一个numpy数组。
    :param result:模型预测结果独热编码矩阵，是一个numpy数组。
    :return:
    """
    cols = y_test.shape[1]
    y = []
    pre = []

    labels = ['setosa', 'versicolor', 'virginica']

    y = [labels[np.argmax(y_test[:, i])] for i in range(y_test.shape[1])]

    pre = [labels[np.argmax(result[:, i])] if np.max(result[:, i]) > 0.5 else 'unknown' for i in range(result.shape[1])]

    y = pd.Series(y)
    pre = pd.Series(pre)


    real = np.concatenate((x_test.T, np.array(y).reshape(-1, 1)), axis=1)
    prediction = np.concatenate((x_test.T, np.array(pre).reshape(-1, 1)), axis=1)

    df_real = pd.DataFrame(real, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    df_prediction = pd.DataFrame(prediction,
                                 columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])

    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[
        ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_prediction[
        ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'], ax=axes[0])
    axes[0].set_title('真实分类')

    radviz(df_prediction, 'Species', color=['blue', 'green', 'red', 'yellow'], ax=axes[1])
    axes[1].set_title('预测分类')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    training_data = pd.read_csv('iris_training.csv', header=None)

    X_train = training_data.iloc[:, :4].values.T
    Y_train = training_data.iloc[:, 4:].values.T
    Y_train = Y_train.astype('uint8')

    start_time = datetime.datetime.now()
    parameters, print_cost, cost_history = nn_model(X_train, Y_train, n_h=10, n_input=4, n_output=3,
                                                    num_iterations=10000,
                                                    print_cost=True)
    end_time = datetime.datetime.now()

    print("训练用时：" + str((end_time - start_time).seconds) + 's' + str(
        round((end_time - start_time).microseconds / 1000)) + 'ms')

    if print_cost:
        plot_cost_history(cost_history)

    test_data = pd.read_csv('iris_test.csv', header=None)
    X_test = test_data.iloc[:, :4].values.T
    Y_test = test_data.iloc[:, 4:].values.T
    Y_test = Y_test.astype('uint8')

    result = predict(parameters, X_test, Y_test)

    result_visualization(X_test, Y_test, result)
