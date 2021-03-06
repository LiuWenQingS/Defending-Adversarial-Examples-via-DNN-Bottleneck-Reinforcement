import argparse
import os
import time
import torch
import torchvision
import random
import numpy as np
from torch import nn, optim
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchsummary import summary
from models import resnet
from torch.utils.data import DataLoader


def train_cla():
    torch.cuda.manual_seed(10)
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    parser = argparse.ArgumentParser("Image classifical!")
    parser.add_argument("--input_path", type=str, default="D:/ETUDE/inputData/mnist/",
                        help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--epochs", type=int, default=300, help="Epoch default:50.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.0001")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--model_path", type=str,
                        default="D:/ETUDE/checkpoint/Classification/ResNet18/mnist/model/",
                        help="Save model path")
    parser.add_argument("--acc_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Classification/ResNet18/mnist/acc.txt",
                        help="Save accuracy file")
    parser.add_argument("--best_acc_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Classification/ResNet18/mnist/best_acc.txt",
                        help="Save best accuracy file")
    parser.add_argument("--log_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Classification/ResNet18/mnist/log.txt",
                        help="Save log file")

    args = parser.parse_args()

    # # Create model
    # if not os.path.exists(args.model_path):
    #     os.makedirs(args.model_path)

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        # transforms.Resize(32),  # 将图像转化为32 * 32
        # transforms.RandomCrop(28, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load data
    train_datasets = torchvision.datasets.MNIST(root=args.input_path,
                                                transform=transform_train,
                                                download=True,
                                                train=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_datasets = torchvision.datasets.MNIST(root=args.input_path,
                                               transform=transform_test,
                                               download=True,
                                               train=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    # Load model
    # if torch.cuda.is_available():
    #     model = torch.load(args.model_path + args.model_name).to(device)
    # else:
    #     model = torch.load(args.model_path + args.model_name, map_location="cpu")
    model = resnet.resnet18(pretrained=False)
    model.to(device)
    # summary(model,(1, 28, 28))
    # print(model)
    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    length = len(train_loader)  # iter数量
    best_acc = 0  # 初始化best test accuracy
    best_epoch = 1  # 初始化best epoch
    print("Start Training, Resnet-18!")
    with open(args.acc_file_path, "w") as f1:
        with open(args.log_file_path, "w")as f2:
            for epoch in range(0, args.epochs):
                if epoch + 1 <= 100:
                    args.lr = 0.1
                elif 100 < epoch + 1 <= 200:
                    args.lr = 0.01
                elif 200 < epoch + 1 <= 250:
                    args.lr = 0.001
                else:
                    args.lr = 0.0001

                # Optimization
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

                print("Epoch: %d" % (epoch + 1))
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    start = time.time()

                    # 准备数据
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    # if len(labels) != 256:
                    #     break
                    # model.noise = nn.Parameter(data=torch.zeros([len(labels), 1, 28, 28]), requires_grad=False)
                    model.to(device)
                    model.train()
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum().item()
                    # print(100.* correct / total)

                    end = time.time()

                    print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                          % (epoch + 1, args.epochs, i + 1, length, sum_loss / (i + 1), correct / total * 100, args.lr,
                             (end - start)))
                    f2.write("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.4f | Time: %.3fs"
                          % (epoch + 1, args.epochs, i + 1, length, sum_loss / (i + 1), correct / total * 100, args.lr,
                             (end - start)))
                    f2.write("\n")
                    f2.flush()
                # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
                # 每训练完一个epoch测试一下准确率
                print("Waiting for Testing!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        model.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        # model.noise = nn.Parameter(data=torch.zeros([len(labels), 1, 28, 28]), requires_grad=False)
                        model.to(device)
                        outputs = model(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
                    print("Test Set Accuracy：%.2f%%" % (correct / total * 100))
                    acc = correct / total * 100
                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Accuracy= %.2f%%" % (epoch + 1, acc))
                    f1.write("\n")
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                    if acc > best_acc:
                        if epoch != 0:
                           os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                        best_acc = acc
                        print("Saving model!")
                        torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch + 1))
                        print("Model saved!")
                        f3 = open(args.best_acc_file_path, "w")
                        f3.write("Epoch=%d,best_acc= %.2f%%" % (epoch + 1, acc))
                        f3.close()
                        best_epoch = epoch + 1
            print("Training Finished, TotalEpoch = %d, Best Accuracy = %.2f%%" % (args.epochs, best_acc))
    return best_epoch


if __name__ == "__main__":
    train_cla()
