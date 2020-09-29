import numpy as np
import pickle
import time
import argparse
import os
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchsummary import summary
from models import resnet_cifar
from torch.utils.data import DataLoader


def load_images(input_dir, batch_shape_clean, batch_shape_adv):
    image_clean = np.zeros(batch_shape_clean)
    image_adv = np.zeros(batch_shape_adv)
    labels = np.zeros(batch_shape_clean[0])
    idx = 0
    batch_size = batch_shape_adv[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_clean_list = data_dict["images_clean"]
    images_adv_list = data_dict["images_adv"]
    labels_list = data_dict["labels"]

    for j in range(len(images_clean_list)):
        image_clean[idx, :, :, :] = images_clean_list[j]
        image_adv[idx, :, :, :] = images_adv_list[j]
        labels[idx] = labels_list[j]

        idx += 1
        if idx == batch_size:
            yield idx, image_clean, image_adv, labels
            image_clean = np.zeros(batch_shape_clean)
            image_adv = np.zeros(batch_shape_adv)
            labels = np.zeros(batch_shape_clean[0])
            idx = 0

    if idx > 0:
        yield idx, image_clean, image_adv, labels


def reTrain(best_cla_model_path, best_ae_epoch, round, block, ae_training_set, device_used):
    # Device configuration
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    # parser.add_argument("--input_path", type=str, default="D:/python_workplace/resnet-AE/inputData/cifar/cifar10/cifar-10-batches-py/",
    #                     help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--input_path", type=str,
                        default="C:/Users/WenqingLiu/cifar/cifar10/cifar-10-batches-py/",
                        help="data set dir path")
    parser.add_argument("--checkpoint_path_ae", type=str,
                        default="H:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) +
                                "/model/model_" + str(best_ae_epoch) + ".pth",
                        help="Path to checkpoint for ae network.")
    parser.add_argument("--input_dir_trainSet", type=str,
                        default="H:/python_workplace/resnet-AE/outputData/FGSM/cifar10/" + ae_training_set +
                                "/block_" + str(block) + "/round_" + str(round) + "/train/train.pkl",
                        help="data set dir path")
    parser.add_argument("--input_dir_testSet", type=str,
                        default="H:/python_workplace/resnet-AE/outputData/FGSM/cifar10/" + ae_training_set +
                                "/block_" + str(block) + "/round_" + str(round) + "/test/test.pkl",
                        help="data set dir path")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--image_size", type=int, default=32, help="Size of each input images.")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch default:50.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.0001")
    parser.add_argument("--model_path", type=str,
                        default="H:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/RetrainClassification/ResNet18/cifar10/block_" + str(block) +
                                "/round_" + str(round) + "/model/",
                        help="Save model path")
    parser.add_argument("--acc_file_path", type=str,
                        default="H:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/RetrainClassification/ResNet18/cifar10/block_" + str(block) +
                                "/round_" + str(round) + "/acc.txt",
                        help="Save accuracy file")
    parser.add_argument("--best_acc_file_path", type=str,
                        default="H:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/RetrainClassification/ResNet18/cifar10/block_" + str(block) +
                                "/round_" + str(round) + "/best_acc.txt",
                        help="Save best accuracy file")
    parser.add_argument("--log_file_path", type=str,
                        default="H:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/RetrainClassification/ResNet18/cifar10/block_" + str(block) +
                                "/round_" + str(round) + "/log.txt",
                        help="Save log file")

    args = parser.parse_args()


    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load data
    train_datasets = torchvision.datasets.CIFAR10(root=args.input_path,
                                                  transform=transform_train,
                                                  download=True,
                                                  train=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_datasets = torchvision.datasets.CIFAR10(root=args.input_path,
                                                 transform=transform_test,
                                                 download=True,
                                                 train=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    model = resnet_cifar.resnet18(pretrained=False)
    model.to(device)

    pretrained_ae_dict = torch.load(args.checkpoint_path_ae)
    pretrained_cla_dict = torch.load(best_cla_model_path)
    model_dict = model.state_dict()

    pretrained_cla_weight_key = []
    pretrained_cla_weight_value = []
    pretrained_ae_weight_key = []
    pretrained_ae_weight_value = []
    model_weight_key = []
    model_weight_value = []
    for k, v in pretrained_cla_dict.items():
        # print(k)
        pretrained_cla_weight_key.append(k)
        pretrained_cla_weight_value.append(v)

    for k, v in pretrained_ae_dict.items():
        # print(k)
        pretrained_ae_weight_key.append(k)
        pretrained_ae_weight_value.append(v)

    for k, v in model_dict.items():
        # print(k)
        model_weight_key.append(k)
        model_weight_value.append(v)

    new_dict = {}
    for i in range(len(model_weight_key)):
        if i < (6 + 4 * 6 * block + 6 * (block - 1)):
            new_dict[model_weight_key[i]] = pretrained_ae_weight_value[i]
        else:
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    batchShape_clean = [args.batch_size, 3, args.image_size, args.image_size]
    batchShape_adv = [args.batch_size, 3, args.image_size, args.image_size]

    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # length_train = len(train_loader)
    best_acc_test_clean = 0
    best_acc_test_adv = 0
    best_acc_train_clean = 0
    best_acc_train_adv = 0
    best_epoch = 1
    flag_test = 0
    flag_train = 0
    print("Start Training Resnet-18 After AutoEncoder!")
    with open(args.acc_file_path, "w") as f1:
        with open(args.log_file_path, "w")as f2:
            for epoch in range(0, args.epochs):
                if epoch + 1 <= 20:
                    args.lr = 0.01
                elif epoch + 1 > 20 & epoch + 1 <= 40:
                    args.lr = 0.001
                else:
                    args.lr = 0.0001

                # Optimization
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

                # 每个epoch之前测试一下准确率
                print("Waiting for Testing of Test Set!")
                with torch.no_grad():
                    correct_clean_test = 0
                    correct_adv_test = 0
                    total_test = 0
                    for batchSize, images_clean, images_adv, labels in load_images(args.input_dir_testSet,
                                                                                   batchShape_clean,
                                                                                   batchShape_adv):
                        model.eval()
                        images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                        images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                        labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
                        model.to(device)
                        total_test += batchSize
                        # 测试clean数据集的测试集
                        outputs_clean = model(images_clean)
                        _, predicted_clean = torch.max(outputs_clean.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                        correct_clean_test += (predicted_clean == labels).sum().item()
                        # 测试adv数据集的测试集
                        outputs_adv = model(images_adv)
                        _, predicted_adv = torch.max(outputs_adv.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                        correct_adv_test += (predicted_adv == labels).sum().item()
                    # print(total_test)
                    acc_clean_test = correct_clean_test / total_test * 100
                    acc_adv_test = correct_adv_test / total_test * 100
                    print("Clean Test Set Accuracy：%.2f%%" % acc_clean_test)
                    print("Adv Test Set Accuracy：%.2f%%" % acc_adv_test)
                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Clean Test Set Accuracy= %.2f%%,Adv Test Set Accuracy = %.2f%%" % (
                        epoch, acc_clean_test, acc_adv_test))
                    f1.write("\n")
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                    if acc_clean_test > best_acc_test_clean:
                        best_acc_test_clean = acc_clean_test
                    if acc_adv_test > best_acc_test_adv:
                        flag_test = 1
                        best_acc_test_adv = acc_adv_test

                print("Waiting for Testing of Train Set!")
                with torch.no_grad():
                    correct_clean_train = 0
                    correct_adv_train = 0
                    total_train = 0
                    for batchSize, images_clean, images_adv, labels in load_images(args.input_dir_trainSet,
                                                                                   batchShape_clean,
                                                                                   batchShape_adv):
                        model.eval()
                        images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                        images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                        labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
                        model.to(device)
                        total_train += batchSize
                        # 测试clean数据集的测试集
                        outputs_clean = model(images_clean)
                        _, predicted_clean = torch.max(outputs_clean.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                        correct_clean_train += (predicted_clean == labels).sum().item()
                        # 测试adv数据集的测试集
                        outputs_adv = model(images_adv)
                        _, predicted_adv = torch.max(outputs_adv.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                        correct_adv_train += (predicted_adv == labels).sum().item()
                    # print(total_train)
                    acc_clean_train = correct_clean_train / total_train * 100
                    acc_adv_train = correct_adv_train / total_train * 100
                    print("Clean Train Set Accuracy：%.2f%%" % acc_clean_train)
                    print("Adv Train Set Accuracy：%.2f%%" % acc_adv_train)
                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Clean Train Set Accuracy= %.2f%%,Adv Train Set Accuracy = %.2f%%" % (
                        epoch, acc_clean_train, acc_adv_train))
                    f1.write("\n")
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                    if acc_clean_train > best_acc_train_clean:
                        best_acc_train_clean = acc_clean_train
                    if acc_adv_train > best_acc_train_adv:
                        flag_train = 1
                        best_acc_train_adv = acc_adv_train

                if flag_train == 1 and flag_test == 1:
                    if epoch != 0:
                        os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                    f3 = open(args.best_acc_file_path, "w")
                    f3.write("Epoch=%03d,Clean Test Set Accuracy= %.2f%%,Adv Test Set Accuracy = %.2f%%,"
                             "Clean Train Set Accuracy= %.2f%%,Adv Train Set Accuracy = %.2f%%"
                             % (epoch, acc_clean_test, acc_adv_test, acc_clean_train, acc_adv_train))
                    f3.close()

                    print("Saving model!")
                    torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch))
                    print("Model saved!")
                    best_epoch = epoch

                flag_test = 0
                flag_train = 0

                print("Epoch: %d" % (epoch + 1))
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                start = time.time()
                batch = 1
                len_batch = len(train_loader)
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    s = time.time()
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
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
                    e = time.time()
                    # print(100.* correct / total)
                    print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                          % (epoch + 1, args.epochs, batch, len_batch, sum_loss / batch, correct / total * 100, args.lr,
                             (e - s)))
                    batch += 1
                end = time.time()

                print("[Epoch:%d/%d] | Loss: %.03f | Test Acc of Clean Test Set: %.2f%% | "
                      "Test Acc of Clean Train Set: %.2f%% | Test Acc of Adv Test Set: %.2f%% | "
                      "Test Acc of Adv Train Set: %.2f%% | Train Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                      % (epoch + 1, args.epochs, sum_loss / (i + 1), acc_clean_test, acc_clean_train,
                         acc_adv_test, acc_adv_train, correct / total * 100, args.lr, (end - start)))
                f2.write("[Epoch:%d/%d] | Loss: %.03f | Test Acc of Clean Test Set: %.2f%% | "
                         "Test Acc of Clean Train Set: %.2f%% | Test Acc of Adv Test Set: %.2f%% | "
                         "Test Acc of Adv Train Set: %.2f%% | Train Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                         % (epoch + 1, args.epochs, sum_loss / (i + 1), acc_clean_test, acc_clean_train,
                            acc_adv_test, acc_adv_train, correct / total * 100, args.lr, (end - start)))
                f2.write("\n")
                f2.flush()
    return best_epoch


if __name__ == "__main__":
    reTrain("D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_282.pth",
            97, 1, 4, "AE_CleanSet", "cuda:0")
