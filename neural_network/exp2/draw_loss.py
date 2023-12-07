import numpy as np
from matplotlib import pyplot as plt


def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"
    return np.asfarray(data, float)


if __name__ == "__main__":
    test_loss_path = r"D:\sukkart\my projects\neural_network\exp2\test_loss.txt"

    y_test_loss = data_read(test_loss_path)
    x_test_loss = range(len(y_test_loss))

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_test_loss为横坐标，y_test_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_test_loss, y_test_loss, color="red", linewidth=1, label="test loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('epoch_test_loss.jpg')
    plt.show()
