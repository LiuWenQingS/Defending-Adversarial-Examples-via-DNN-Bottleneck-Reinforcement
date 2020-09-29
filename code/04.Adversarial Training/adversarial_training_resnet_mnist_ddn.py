import argparse
import os
import time
import torch
from torch import nn, optim
from torchsummary import summary
from models import resnet
import random
import numpy as np
import pickle

randnum_train = random.randint(0, 100)
randnum_test = random.randint(0, 100)


def load_train_set(input_dir, batch_shape):
    image_train = np.zeros(batch_shape)
    labels_train = np.zeros(batch_shape[0])
    idx = 0
    batch_size = batch_shape[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_list = data_dict["images_clean"]  # clean image
    images_adv_list = data_dict["images_adv"]  # adv image
    labels_list = data_dict["labels"]
    images_list.extend(images_adv_list)
    labels_list.extend(labels_list)

    random.seed(randnum_train)
    random.shuffle(images_list)
    random.seed(randnum_train)
    random.shuffle(labels_list)

    for j in range(len(images_list)):
        image_train[idx, 0, :, :] = images_list[j]
        labels_train[idx] = labels_list[j]

        idx += 1
        if idx == batch_size:
            yield idx, image_train, labels_train
            image_train = np.zeros(batch_shape)
            labels_train = np.zeros(batch_shape[0])
            idx = 0

    if idx > 0:
        yield idx, image_train, labels_train


def load_test_set_clean(input_dir, batch_shape):
    image_test_clean = np.zeros(batch_shape)
    labels_test_clean = np.zeros(batch_shape[0])
    idx = 0
    batch_size = batch_shape[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_clean_list = data_dict["images_clean"]
    # images_adv_list = data_dict["images_adv"]
    labels_list = data_dict["labels"]

    for j in range(len(images_clean_list)):
        image_test_clean[idx, 0, :, :] = images_clean_list[j]
        labels_test_clean[idx] = labels_list[j]

        idx += 1
        if idx == batch_size:
            yield idx, image_test_clean, labels_test_clean
            image_test_clean = np.zeros(batch_shape)
            labels_test_clean = np.zeros(batch_shape[0])
            idx = 0

    if idx > 0:
        yield idx, image_test_clean, labels_test_clean


def load_test_set_adv(input_dir, batch_shape):
    image_test_adv = np.zeros(batch_shape)
    labels_test_adv = np.zeros(batch_shape[0])
    idx = 0
    batch_size = batch_shape[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    # images_clean_list = data_dict["images_clean"]
    images_adv_list = data_dict["images_adv"]
    labels_list = data_dict["labels"]

    for j in range(len(images_adv_list)):
        image_test_adv[idx, 0, :, :] = images_adv_list[j]
        labels_test_adv[idx] = labels_list[j]

        idx += 1
        if idx == batch_size:
            yield idx, image_test_adv, labels_test_adv
            image_test_adv = np.zeros(batch_shape)
            labels_test_adv = np.zeros(batch_shape[0])
            idx = 0

    if idx > 0:
        yield idx, image_test_adv, labels_test_adv


def adversarial_learning(best_cla_model_path):
    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # print(device)

    parser = argparse.ArgumentParser("Image classifical!")
    parser.add_argument('--input_dir_trainSet', type=str,
                        default='D:/python_workplace/resnet-AE/outputData/DDN/mnist/train/train.pkl',
                        help='data set dir path')
    parser.add_argument('--input_dir_testSet', type=str,
                        default='D:/python_workplace/resnet-AE/outputData/DDN/mnist/test/test.pkl',
                        help='data set dir path')
    parser.add_argument('--epochs', type=int, default=300, help='Epoch default:50.')
    parser.add_argument('--image_size', type=int, default=28, help='Image Size default:28.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch_size default:256.')
    parser.add_argument('--lr', type=float, default=0.01, help='learing_rate. Default=0.01')
    parser.add_argument('--num_classes', type=int, default=10, help='num classes')
    parser.add_argument('--model_path', type=str,
                        default='D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/DDN/model/',
                        help='Save model path')
    parser.add_argument('--acc_file_path', type=str,
                        default='D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/DDN/acc.txt',
                        help='Save accuracy file')
    parser.add_argument('--best_acc_file_path', type=str,
                        default='D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/DDN/best_acc.txt',
                        help='Save best accuracy file')
    parser.add_argument('--log_file_path', type=str,
                        default='D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/DDN/log.txt',
                        help='Save log file')

    args = parser.parse_args()

    # Load model
    model = resnet.resnet18(pretrained=False)
    model.to(device)
    # summary(model,(3,32,32))
    # print(model)

    # Load pre-trained weights
    # model.load_state_dict(torch.load(best_cla_model_path))
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # batch_shape
    batch_shape = [args.batch_size, 1, args.image_size, args.image_size]

    best_acc_clean = 50  # 初始化best clean test set accuracy
    best_acc_adv = 50  # 初始化best adv test set accuracy
    best_epoch = 0  # 初始化best epoch
    time_k = time.time()
    print("Start Adversarial Training, Resnet-18!")
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

                print('Epoch: %d' % (epoch + 1))
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                batchId = 1
                for batchSize, images_train, labels_train in load_train_set(args.input_dir_trainSet, batch_shape):
                    start = time.time()

                    # data prepare
                    images_train = torch.from_numpy(images_train).type(torch.FloatTensor).to(device)
                    labels_train = torch.from_numpy(labels_train).type(torch.LongTensor).to(device)

                    # model.noise = nn.Parameter(data=torch.zeros([len(images_train), 1, 28, 28]), requires_grad=False)
                    model.to(device)
                    model.train()
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = model(images_train)
                    loss = criterion(outputs, labels_train)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels_train.size(0)
                    correct += predicted.eq(labels_train.data).cpu().sum().item()
                    # print(100.* correct / total)

                    end = time.time()

                    print('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs'
                          % (epoch + 1, args.epochs, batchId, (120000 / args.batch_size) + 1, sum_loss / batchId,
                             correct / total * 100, args.lr, (end - start)))
                    f2.write('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.4f | Time: %.3fs'
                          % (epoch + 1, args.epochs, batchId, (120000 / args.batch_size) + 1, sum_loss / batchId,
                             correct / total * 100, args.lr, (end - start)))
                    f2.write('\n')
                    f2.flush()
                    batchId += 1
                # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)

                # 每训练完一个epoch测试一下准确率
                print("Waiting for Testing!")
                with torch.no_grad():
                    # 测试clean test set
                    correct_clean = 0
                    total_clean = 0
                    for batchSize, images_test_clean, labels_test_clean in load_test_set_clean(args.input_dir_testSet,
                                                                                               batch_shape):
                        model.eval()

                        # data prepare
                        images_test_clean = torch.from_numpy(images_test_clean).type(torch.FloatTensor).to(device)
                        labels_test_clean = torch.from_numpy(labels_test_clean).type(torch.LongTensor).to(device)

                        # model.noise = nn.Parameter(data=torch.zeros([len(labels_test_clean), 1, 28, 28]), requires_grad=False)
                        model.to(device)

                        outputs = model(images_test_clean)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total_clean += labels_test_clean.size(0)
                        correct_clean += (predicted == labels_test_clean).sum().item()
                    # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
                    print('Clean Test Set Accuracy：%.2f%%' % (correct_clean / total_clean * 100))
                    acc_clean = correct_clean / total_clean * 100

                    # 测试adv test set
                    correct_adv = 0
                    total_adv = 0
                    for batchSize, images_test_adv, labels_test_adv in load_test_set_adv(args.input_dir_testSet,
                                                                                               batch_shape):
                        model.eval()

                        # data prepare
                        images_test_adv = torch.from_numpy(images_test_adv).type(torch.FloatTensor).to(device)
                        labels_test_adv = torch.from_numpy(labels_test_adv).type(torch.LongTensor).to(device)

                        # model.noise = nn.Parameter(data=torch.zeros([len(labels_test_adv), 1, 28, 28]), requires_grad=False)
                        model.to(device)

                        outputs = model(images_test_adv)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total_adv += labels_test_adv.size(0)
                        correct_adv += (predicted == labels_test_adv).sum().item()
                    # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
                    print('Adv Test Set Accuracy：%.2f%%' % (correct_adv / total_adv * 100))
                    acc_adv = correct_adv / total_adv * 100

                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Clean Test Set Accuracy= %.2f%%" % (epoch + 1, acc_clean))
                    f1.write('\n')
                    f1.write("Epoch=%03d,Adv Test Set Accuracy= %.2f%%" % (epoch + 1, acc_adv))
                    f1.write('\n')
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                    if acc_clean > best_acc_clean and acc_adv > best_acc_adv:
                        if epoch != 0:
                           os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                        best_acc_clean = acc_clean
                        best_acc_adv = acc_adv
                        print('Saving model!')
                        torch.save(model.state_dict(), '%s/model_%d.pth' % (args.model_path, epoch + 1))
                        print('Model saved!')
                        f3 = open(args.best_acc_file_path, "w")
                        f3.write("Epoch=%d,Best Accuracy of Clean Set = %.2f%%,Best Accuracy of Adv Set = %.2f%%"
                                 % (epoch + 1, best_acc_clean, best_acc_adv))
                        f3.close()
                        best_epoch = epoch + 1
            time_j = time.time()
            print("Training Finished, Total Epoch = %d, Best Epoch = %d, Best Accuracy of Clean Set = %.2f%%, "
                  "Best Accuracy of Adv Set = %.2f%%, Total Time = %.2f" % (args.epochs, best_epoch, best_acc_clean,
                                                                            best_acc_adv, (time_j - time_k)/3600))


if __name__ == '__main__':
    best_cla_model_path = 'D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_' + str(
        69) + '.pth'
    adversarial_learning(best_cla_model_path)