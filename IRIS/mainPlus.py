import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
from pandas.plotting import radviz

'''
    构建一个具有1个隐藏层的神经网络，隐层的大小为10
    输入层为4个特征，输出层为3个分类
    (1,0,0)为第一类，(0,1,0)为第二类，(0,0,1)为第三类
'''


# 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x:输入层的节点数
    :param n_h:隐层的节点数
    :param n_y:输出层的节点数
    :return:parameters
    """
    # seed用于指定随机数生成器的种子，也称为随机数生成器的“状态”，其目的是为了使随机数的生成可重复。
    # 如果不指定随机数生成器的种子，每次生成的随机数序列都会不同。
    # 指定了种子为2，这意味着每次运行该代码，生成的随机数序列将始终相同，
    # 这是为了确保结果的可重复性，方便调试和验证算法的正确性。
    np.random.seed(2)

    # 初始化权重和偏置矩阵，使用Xavier初始化
    # Xavier初始化可以更好地使得网络中每一层的梯度都有相同的方差，有助于提高网络的训练效果

    # 使用Xavier初始化方法初始化权重矩阵w1和w2
    w1 = np.random.randn(n_h, n_x) * np.sqrt(2 / n_x)
    w2 = np.random.randn(n_y, n_h) * np.sqrt(2 / n_h)
    # 初始化偏置向量b1和b2为零向量
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    # 将初始化好的参数存储到字典中
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    # 返回参数字典
    return parameters


# 将X和参数进行前向传播计算，得到预测值和缓存的中间结果
def forward_propagation(X, parameters):
    """
    :param X:数据集X
    :param parameters:包含神经网络参数的字典parameters
    :return:数组a2，包含神经网络的输出结果；字典cache，包含前向传播时计算过程中的一些中间变量，在反向传播时使用。
    """
    # 从参数中提取出w1、b1、w2和b2
    w1, b1, w2, b2 = parameters['w1'], parameters['b1'], parameters['w2'], parameters['b2']
    # 计算z1、a1、z2和a2
    z1 = np.dot(w1, X) + b1
    # 使用tanh作为第一层的激活函数
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    # 使用sigmoid作为第二层的激活函数
    a2 = 1 / (1 + np.exp(-z2))
    # 将中间结果存储在缓存中
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


# 计算代价函数
def compute_cost(a2, Y, parameters, lambd=0.3):
    """
    :param a2:预测值，shape为(1, m)，其中m为样本数
    :param Y:实际标签，shape为(1, m)
    :param parameters:包含权重矩阵w1和w2的字典
    :param lambd: L2正则化超参数
    :return:cost: 代价函数的值
    """
    # 样本数
    m = Y.shape[1]
    # 计算交叉熵代价函数
    log_probs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cross_entropy_cost = - np.sum(log_probs) / m
    # 从参数字典中获取权重矩阵
    w1, w2 = parameters['w1'], parameters['w2']
    # 添加L2正则化
    l2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    # 总代价函数
    cost = cross_entropy_cost + l2_regularization_cost
    # 返回代价函数的值
    return cost


# 反向传播（计算神经网络的梯度值）
def backward_propagation(parameters, cache, X, Y, lambd=0.3):
    """
    :param parameters:包含模型参数 w1, b1, w2, b2 的字典
    :param cache:包含中间结果的字典，包括 a1, a2
    :param X:输入特征矩阵
    :param Y:标签向量
    :param lambd:L2正则化系数
    :return:grads：包含梯度 dw1, db1, dw2, db2 的字典
    """
    # 样本数量
    m = Y.shape[1]
    # 从参数字典和缓存中获取权重和中间结果
    w1, w2, a1, a2 = parameters['w1'], parameters['w2'], cache['a1'], cache['a2']
    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = np.dot(dz2, a1.T) / m + (lambd / m) * w2
    db2 = np.mean(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = np.dot(dz1, X.T) / m + (lambd / m) * w1
    db1 = np.mean(dz1, axis=1, keepdims=True)
    # 将梯度存入字典
    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


# 使用Adam优化算法来更新神经网络的参数，参数包括权重和偏置项
def update_parameters_with_adam(parameters, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    :param parameters:包含神经网络所有参数的字典，其中键是参数名，值是参数的numpy数组
    :param grads:包含神经网络所有参数梯度的字典，其中键是参数名，值是参数的梯度numpy数组
    :param learning_rate:学习率，控制参数更新的速度，默认值为0.01
    :param beta1:第一次指数加权平均值的权重，控制动量项的权重，默认值为0.9
    :param beta2:第二次指数加权平均值的权重，控制RMSProp项的权重，默认值为0.999
    :param epsilon:避免除零错误的小值，默认值为1e-8
    :return:parameters: 更新后的参数字典，其中键是参数名，值是参数的numpy数组
    """
    # 从参数字典中获取权重和偏置项
    w1, b1, w2, b2 = parameters.values()
    # 从梯度字典中获取梯度
    dw1, db1, dw2, db2 = grads.values()
    # 初始化动量向量和RMSProp指数加权平均值
    vdW1, vdW2 = np.zeros_like(w1), np.zeros_like(w2)
    sdW1, sdW2 = np.zeros_like(w1), np.zeros_like(w2)
    vdb1, vdb2 = np.zeros_like(b1), np.zeros_like(b2)
    sdb1, sdb2 = np.zeros_like(b1), np.zeros_like(b2)
    # 计算动量向量和RMSProp指数加权平均值
    vdW1 = beta1 * vdW1 + (1 - beta1) * dw1
    vdb1 = beta1 * vdb1 + (1 - beta1) * db1
    vdW2 = beta1 * vdW2 + (1 - beta1) * dw2
    vdb2 = beta1 * vdb2 + (1 - beta1) * db2
    sdW1 = beta2 * sdW1 + (1 - beta2) * np.square(dw1)
    sdb1 = beta2 * sdb1 + (1 - beta2) * np.square(db1)
    sdW2 = beta2 * sdW2 + (1 - beta2) * np.square(dw2)
    sdb2 = beta2 * sdb2 + (1 - beta2) * np.square(db2)
    # 根据动量向量和RMSProp指数加权平均值来更新权重和偏置项
    w1 -= (learning_rate * vdW1) / (np.sqrt(sdW1) + epsilon)
    b1 -= (learning_rate * vdb1) / (np.sqrt(sdb1) + epsilon)
    w2 -= (learning_rate * vdW2) / (np.sqrt(sdW2) + epsilon)
    b2 -= (learning_rate * vdb2) / (np.sqrt(sdb2) + epsilon)
    # 将更新后的参数存入字典中
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


# 模型评估
def predict(parameters, x_test, y_test):
    """
    :param parameters: 包含训练好的神经网络参数的字典
    :param x_test:测试集特征矩阵，大小为(n_x, m)
    :param y_test:测试集标签矩阵，大小为(n_y, m)
    :return:output为预测结果，大小为(1, m)
    """
    w1 = parameters['w1']  # 第一层神经网络的权重矩阵，大小为(n_h, n_x)
    b1 = parameters['b1']  # 第一层神经网络的偏置矩阵，大小为(n_h, 1)
    w2 = parameters['w2']  # 第二层神经网络的权重矩阵，大小为(n_y, n_h)
    b2 = parameters['b2']  # 第二层神经网络的偏置矩阵，大小为(n_y, 1)
    z1 = np.dot(w1, x_test) + b1  # 第一层神经网络的加权输入，大小为(n_h, m)
    a1 = np.tanh(z1)  # 第一层神经网络的输出，大小为(n_h, m)
    z2 = np.dot(w2, a1) + b2  # 第二层神经网络的加权输入，大小为(n_y, m)
    a2 = 1 / (1 + np.exp(-z2))  # 第二层神经网络的输出，大小为(n_y, m)
    # 预测结果，大小为(1, m)
    output = np.where(a2 > 0.5, 1, 0)  # 使用numpy中的where函数替代for循环
    print('预测结果：')
    print(output)
    print("\n")
    print('真实结果：')
    print(y_test)
    # 预测准确率，单位为百分比
    accuracy = np.mean(np.all(output == y_test, axis=0)) * 100  # 使用numpy中的mean和all函数计算准确率
    print('准确率：%.2f%%' % accuracy)
    return output


# 建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations=10000, print_cost=False):
    """
    :param X:接收输入数据
    :param Y:标签
    :param n_h:隐层节点数
    :param n_input:输入层节点数
    :param n_output:输出层节点数
    :param num_iterations:迭代次数
    :param print_cost:是否输出代价函数的标志
    :return:更新后的参数parameters、是否输出代价函数的标志print_cost和代价函数历史值cost_history
    """
    # 设置随机数种子为3，以保证可复现性
    np.random.seed(3)
    # 获取输入层、隐层和输出层节点数
    n_x = n_input  # 输入层节点数
    n_y = n_output  # 输出层节点数
    # 初始化神经网络的参数，包括权重和偏置，其中隐层节点数为n_h
    parameters = initialize_parameters(n_x, n_h, n_y)
    # 初始化代价函数历史值
    cost_history = []
    # 梯度下降循环迭代神经网络模型，共进行num_iterations次迭代
    for i in range(1, num_iterations + 1):
        # 进行前向传播，得到输出层的输出a2和存储中间结果的cache
        a2, cache = forward_propagation(X, parameters)
        # 计算代价函数
        cost = compute_cost(a2, Y, parameters)
        # 进行反向传播，得到梯度信息
        grads = backward_propagation(parameters, cache, X, Y)
        # 根据梯度信息更新参数
        parameters = update_parameters_with_adam(parameters, grads)
        # 保存代价函数历史值
        if i % 100 == 0:
            cost_history.append(cost)
        # 每1000次迭代，输出一次代价函数
        if print_cost and i % 1000 == 0:
            print('迭代第%i次     代价函数：%f' % (i, cost))
            print("-----------------------------------------------")
    return parameters, print_cost, cost_history


# 绘制神经网络模型的代价函数历史值随迭代次数变化的曲线图
def plot_cost_history(cost_history):
    """
    :param cost_history:包含每次迭代的代价函数历史值的列表
    :return:NULL
    """
    plt.figure('代价函数')
    plt.plot(cost_history)
    plt.title('Cost Function')
    plt.xlabel('Iterations (per 100)')
    plt.ylabel('Cost')
    plt.show()


# 结果可视化
# 特征有4个维度，类别有1个维度，一共5个维度，故采用了RadViz图
def result_visualization(x_test, y_test, result):
    """
    :param x_test:测试集特征矩阵，是一个numpy数组。
    :param y_test:测试集标签独热编码矩阵，是一个numpy数组。
    :param result:模型预测结果独热编码矩阵，是一个numpy数组。
    :return:
    """
    cols = y_test.shape[1]  # 获取测试集的列数，即样本数
    y = []  # 存储反转换后的真实分类
    pre = []  # 存储反转换后的预测分类
    # 反转换类别的独热编码
    # 定义标签的名称
    labels = ['setosa', 'versicolor', 'virginica']
    # 反转换测试集
    y = [labels[np.argmax(y_test[:, i])] for i in range(y_test.shape[1])]
    # 反转换预测结果
    pre = [labels[np.argmax(result[:, i])] if np.max(result[:, i]) > 0.5 else 'unknown' for i in range(result.shape[1])]
    # 将y和pre转换为pandas的Series对象
    y = pd.Series(y)
    pre = pd.Series(pre)
    # 特征矩阵拼接
    # 将特征和类别矩阵拼接起来
    real = np.concatenate((x_test.T, np.array(y).reshape(-1, 1)), axis=1)
    prediction = np.concatenate((x_test.T, np.array(pre).reshape(-1, 1)), axis=1)
    # 转换成DataFrame类型，并添加columns
    df_real = pd.DataFrame(real, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    df_prediction = pd.DataFrame(prediction,
                                 columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    # 将特征列转换为float类型，否则radviz会报错
    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[
        ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_prediction[
        ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 绘制真实分类
    radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'], ax=axes[0])
    axes[0].set_title('真实分类')
    # 绘制预测分类
    radviz(df_prediction, 'Species', color=['blue', 'green', 'red', 'yellow'], ax=axes[1])
    axes[1].set_title('预测分类')
    # 调整子图之间的间距和外边距
    plt.tight_layout()
    # 显示图像
    plt.show()


if __name__ == "__main__":
    # 读取训练数据
    training_data = pd.read_csv('iris_training.csv', header=None)
    # 获取特征和标签
    X_train = training_data.iloc[:, :4].values.T
    Y_train = training_data.iloc[:, 4:].values.T
    Y_train = Y_train.astype('uint8')
    # 训练模型
    start_time = datetime.datetime.now()
    parameters, print_cost, cost_history = nn_model(X_train, Y_train, n_h=10, n_input=4, n_output=3,
                                                    num_iterations=10000,
                                                    print_cost=True)
    end_time = datetime.datetime.now()
    # 输出训练用时
    print("训练用时：" + str((end_time - start_time).seconds) + 's' + str(
        round((end_time - start_time).microseconds / 1000)) + 'ms')
    # 绘制代价函数曲线
    if print_cost:
        plot_cost_history(cost_history)
    # 对模型进行测试
    test_data = pd.read_csv('iris_test.csv', header=None)
    X_test = test_data.iloc[:, :4].values.T
    Y_test = test_data.iloc[:, 4:].values.T
    Y_test = Y_test.astype('uint8')
    # 预测结果
    result = predict(parameters, X_test, Y_test)
    # 分类结果可视化
    result_visualization(X_test, Y_test, result)
