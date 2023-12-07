import torch
from torch import nn
import torchvision
from tqdm import tqdm
import csv

Batch_size_list = [16, 32, 64, 128, 256, 512]
learning_rate_list = [1e-3, 1e-4, 1e-5, 1e-6]
keep_prob_rate_list = [0.5, 0.6, 0.7, 0.8]

Epoches = 26
Batch_size = 100
learning_rate = 1e-4
keep_prob_rate = 0.7

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
)

path = "./data/"
trainData = torchvision.datasets.MNIST(
    path, train=True, transform=transform, download=True
)

testData = torchvision.datasets.MNIST(path, train=False, transform=transform)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=7, padding=3, stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=5, padding=2, stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(1 - keep_prob_rate),
            torch.nn.Linear(in_features=1024, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target
    # print(data.shape)
    # print(target.shape)

    # print("the value of current Epoch is :", Epoches)
    params = ['Batch_size', 'learning_rate', 'keep_prob_rate', 'loss', 'accuracy']
    with open('params.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Batch_size', 'learning_rate', 'keep_prob_rate', 'loss', 'accuracy'])
        writer.writerow(params)
    for Batch_size in Batch_size_list:
        for learning_rate in learning_rate_list:
            for keep_prob_rate in keep_prob_rate_list:
                params = []
                params.append(Batch_size)
                params.append(learning_rate)
                params.append(keep_prob_rate)
                trainDataLoader = torch.utils.data.DataLoader(
                    dataset=trainData, batch_size=Batch_size, shuffle=True
                )
                testDataLoader = torch.utils.data.DataLoader(
                    dataset=testData, batch_size=Batch_size
                )
                net = Net()
                print(net.to(device))
                lossF = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
                history = {'Test Loss': [], 'Test Accuracy': []}
                test_loss = 0.0
                test_acc = 0.0
                for epoch in range(1, Epoches + 1):
                    processBar = tqdm(trainDataLoader, unit='step')
                    net.train(True)
                    for step, (trainImgs, labels) in enumerate(processBar):
                        trainImgs = trainImgs.to(device)
                        labels = labels.to(device)
                        net.zero_grad()
                        outputs = net(trainImgs)
                        loss = lossF(outputs, labels)
                        predictions = torch.argmax(outputs, dim=1)
                        accuracy = torch.sum(predictions == labels) / labels.shape[0]
                        loss.backward()
                        optimizer.step()
                        processBar.set_description(
                            "[%d/%d] Loss: %.8f, Acc: %.8f" % (
                                epoch, Epoches, loss.item(), accuracy.item()
                            )
                        )
                        if step == len(processBar) - 1:
                            correct, totalLoss = 0, 0
                            net.train(False)
                            for testImgs, labels in testDataLoader:
                                testImgs = testImgs.to(device)
                                labels = labels.to(device)
                                outputs = net(testImgs)
                                loss = lossF(outputs, labels)
                                predictions = torch.argmax(outputs, dim=1)
                                totalLoss += loss
                                correct += torch.sum(predictions == labels)
                            testAccuracy = correct / (Batch_size * len(testDataLoader))
                            testLoss = totalLoss / len(testDataLoader)
                            history['Test Loss'].append(testLoss.item())
                            history['Test Accuracy'].append(testAccuracy.item())
                            processBar.set_description(
                                "[%d/%d] Loss: %.8f, Acc: %.8f, Test Loss: %.8f, Test Acc: %.8f" % (
                                    epoch, Epoches, loss.item(), accuracy.item(),
                                    testLoss.item(), testAccuracy.item()
                                )
                            )
                            # test_loss.append(testLoss.item())
                            # test_acc.append(testAccuracy.item())
                            test_loss = testLoss.item()
                            test_acc = testAccuracy.item()
                    processBar.close()
                print(test_loss)
                print(test_acc)
                params.append(test_loss)
                params.append(test_acc)
                with open('params.csv', 'a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(params)
                print(params)
