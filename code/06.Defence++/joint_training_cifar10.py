import numpy as np
import pickle
import time
import argparse
import os
import torch
import torchvision
import random
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchsummary import summary
from models.joint_training_cifar10 import joint_model_1, joint_model_2, joint_model_3, \
    BasicBlock_encoder, BasicBlock_decoder, joint_model_block2_1, joint_model_block3_1, joint_model_block4_1
from torch.utils.data import DataLoader


torch.cuda.manual_seed(10)
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


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


def joint_train(best_cla_model_path, best_ae_model_path, device_used):
    # Device configuration
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument("--input_path", type=str, default="D:/python_workplace/resnet-AE/inputData/cifar/cifar10/cifar-10-batches-py/",
                        help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--image_size", type=int, default=32, help="Size of each input images.")
    parser.add_argument("--epochs", type=int, default=250, help="Epoch default:50.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.0001")
    parser.add_argument("--input_dir_trainSet", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/train/train.pkl",
                        help="data set dir path")
    parser.add_argument("--input_dir_testSet", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/test/test.pkl",
                        help="data set dir path")
    parser.add_argument("--model_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/model/",
                        help="Save model path")
    parser.add_argument("--image_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/image/",
                        help="Save log file")
    parser.add_argument("--acc_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/acc.txt",
                        help="Save accuracy file")
    parser.add_argument("--best_acc_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/best_acc.txt",
                        help="Save best accuracy file")
    parser.add_argument("--log_cla_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/log_cla.txt",
                        help="Save log file")
    parser.add_argument("--log_ae_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/log_ae.txt",
                        help="Save log file")
    parser.add_argument("--loss_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/loss.txt",
                        help="Save log file")
    parser.add_argument("--min_loss_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_10/min_loss.txt",
                        help="Save log file")

    args = parser.parse_args()


    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
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

    model_1 = joint_model_1(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_2 = joint_model_2(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_3 = joint_model_3(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_4 = joint_model_block2_1(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_5 = joint_model_block3_1(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_6 = joint_model_block4_1(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_list = [model_1, model_2, model_3, model_4, model_5, model_6]
    model = model_list[0]

    pretrained_cla_dict = torch.load(best_cla_model_path)
    pretrained_ae_dict = torch.load(best_ae_model_path)
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
    for i in range(len(model_dict)):
        if i < 30:
        # if i < 60:
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i]
        elif 30 <= i < len(pretrained_ae_weight_key):
            new_dict[model_weight_key[i]] = pretrained_ae_weight_value[i]
            # continue
        else:
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i - len(pretrained_ae_weight_key) + 30]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    # summary(model, (1, 28, 28))

    batchShape_clean = [args.batch_size, 3, args.image_size, args.image_size]
    batchShape_adv = [args.batch_size, 3, args.image_size, args.image_size]

    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    # criterion
    criterion_cla = nn.CrossEntropyLoss().to(device)
    criterion_ae = nn.MSELoss()

    length = len(train_loader)

    # initialize figure
    f, a = plt.subplots(4, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    best_acc_test_adv = 0
    min_loss_clean = 100
    min_loss_adv = 100
    best_epoch = 1
    print("Start Joint Training!")
    with open(args.log_cla_file_path, "w") as f1:
        with open(args.acc_file_path, "w")as f2:
            with open(args.log_ae_file_path, "w")as f3:
                with open(args.loss_file_path, "w")as f4:
                    for epoch in range(0, args.epochs):
                        if epoch < 40:
                            args.lr = 0.01
                        elif epoch >= 40 & epoch < 80:
                            args.lr = 0.001
                        else:
                            args.lr = 0.0001

                        # training
                        print("Training Epoch: %d" % (epoch + 1))
                        sum_loss_cla = 0.0
                        sum_loss_ae = 0.0
                        correct_cla = 0.0
                        total_train = 0
                        batch = 1
                        for batchSize, images_clean, images_adv, labels in load_images(args.input_dir_trainSet,
                                                                                       batchShape_clean,
                                                                                       batchShape_adv):
                            start = time.time()
                            # data prepare
                            images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
                            model.to(device)
                            model.train()

                            # forward + backward
                            outputs_cla, outputs_ae = model(images_clean)
                            if (epoch // 10) % 2 == 1:
                                # Optimization
                                parameters = [{"params":model.conv1.parameters(), "params":model.encoder_layer1.parameters(),
                                               "params":model.layer2.parameters(), "params":model.layer3.parameters(),
                                               "params":model.layer4.parameters(), "params":model.fc.parameters()}]
                                optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
                                optimizer.zero_grad()
                                loss_cla = criterion_cla(outputs_cla, labels)
                                loss_cla.backward()
                                optimizer.step()

                                # classification
                                sum_loss_cla += loss_cla.item()
                                _, predicted_cla = torch.max(outputs_cla.data, 1)
                                total_train += labels.size(0)
                                correct_cla += predicted_cla.eq(labels.data).cpu().sum().item()
                                # print(100.* correct / total_train)

                                end = time.time()

                                print("Log of Classification:")
                                print(
                                    "[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                                    % (epoch + 1, args.epochs, batch, length, sum_loss_cla / batch,
                                       correct_cla / total_train * 100, args.lr, (end - start)))
                                f1.write(
                                    "[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.4f | Time: %.3fs"
                                    % (epoch + 1, args.epochs, batch, length, sum_loss_cla / batch,
                                       correct_cla / total_train * 100, args.lr, (end - start)))
                                f1.write("\n")
                                f1.flush()

                                batch += 1

                            else:
                                # Optimization
                                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                                optimizer.zero_grad()

                                loss_cla = criterion_cla(outputs_cla, labels)
                                loss_ae = criterion_ae(outputs_ae, images_clean)
                                if loss_cla <= 1:
                                    loss = loss_cla + loss_ae
                                else:
                                    loss = loss_cla + 0.8 * loss_ae
                                loss.backward()
                                optimizer.step()

                                # classification
                                sum_loss_cla += loss_cla.item()
                                _, predicted_cla = torch.max(outputs_cla.data, 1)
                                total_train += labels.size(0)
                                correct_cla += predicted_cla.eq(labels.data).cpu().sum().item()
                                # print(100.* correct / total_train)

                                # auto encoder
                                sum_loss_ae += loss_ae.cpu().data.numpy()

                                end = time.time()

                                print("Log of Classification:")
                                print(
                                    "[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                                    % (epoch + 1, args.epochs, batch, length, sum_loss_cla / batch,
                                       correct_cla / total_train * 100, args.lr, (end - start)))
                                f1.write(
                                    "[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.4f | Time: %.3fs"
                                    % (epoch + 1, args.epochs, batch, length, sum_loss_cla / batch,
                                       correct_cla / total_train * 100, args.lr, (end - start)))
                                f1.write("\n")
                                f1.flush()

                                print("Log of AutoEncoder:")
                                print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.04f | Time: %.03fs"
                                      % (epoch + 1, args.epochs, batch, length, sum_loss_ae / batch,
                                         args.lr, (end - start)))
                                f3.write("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs"
                                         % (epoch + 1, args.epochs, batch, length, sum_loss_ae / batch,
                                            args.lr, (end - start)))
                                f3.write("\n")
                                f3.flush()

                                batch += 1

                        if epoch + 1 == 1:
                            print("Saving model!")
                            torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch + 1))
                            print("Model saved!")

                        # testing
                        if (epoch + 1) % 10 == 0:
                            print("Waiting for Testing Cleaning Set!")
                            with torch.no_grad():
                                correct_clean_cla = 0
                                correct_adv_cla = 0
                                total_test = 0
                                sum_loss_clean = 0
                                sum_loss_adv = 0
                                batch_id_test = 1
                                # for data in test_loader:
                                #     model.eval()
                                #     images, labels = data
                                #     images, labels = images.to(device), labels.to(device)
                                #     model.to(device)
                                #     outputs_cla, outputs_ae = model(images)
                                #     # 取得分最高的那个类 (outputs.data的索引号)
                                #     _, predicted = torch.max(outputs_cla.data, 1)
                                #     total_test += labels.size(0)
                                #     correct_clean_cla += (predicted == labels).sum().item()
                                # print(correct_clean_cla, total_test)
                                #
                                # correct_clean_cla = 0
                                # total_test = 0
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
                                    outputs_clean_cla, outputs_clean_ae = model(images_clean)
                                    _, predicted_clean_cla = torch.max(outputs_clean_cla.data, 1)
                                    correct_clean_cla += (predicted_clean_cla == labels).sum().item()

                                    loss_clean = criterion_ae(outputs_clean_ae, images_clean)
                                    sum_loss_clean += loss_clean.cpu().data.numpy()

                                    # 测试adv数据集的测试集
                                    outputs_adv_cla, outputs_adv_ae = model(images_adv)
                                    _, predicted_adv_cla = torch.max(outputs_adv_cla.data, 1)
                                    correct_adv_cla += (predicted_adv_cla == labels).sum().item()

                                    loss_adv = criterion_ae(outputs_adv_ae, images_clean)
                                    sum_loss_adv += loss_adv.cpu().data.numpy()

                                    outputs_clean_ae = torch.clamp(outputs_clean_ae, 0, 1)
                                    outputs_adv_ae = torch.clamp(outputs_adv_ae, 0, 1)

                                    if batch_id_test == 4:
                                        # first row:clean images
                                        for i in range(5):
                                            a[0][i].clear()
                                            a[0][i].imshow(images_clean.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                            a[0][i].set_xticks(())
                                            a[0][i].set_yticks(())
                                            a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")

                                        # second row:adv images
                                        for i in range(5):
                                            a[1][i].clear()
                                            a[1][i].imshow(images_adv.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                            a[1][i].set_xticks(())
                                            a[1][i].set_yticks(())
                                            a[1][i].set_title("Adv Images" + "[" + str(i + 1) + "]")

                                        # third row:decoded clean images
                                        for i in range(5):
                                            a[2][i].clear()
                                            a[2][i].imshow(outputs_clean_ae.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                            a[2][i].set_xticks(())
                                            a[2][i].set_yticks(())
                                            a[2][i].set_title("Clean Images Decoded" + "[" + str(i + 1) + "]")

                                        # fourth row:decoded adv images
                                        for i in range(5):
                                            a[3][i].clear()
                                            a[3][i].imshow(outputs_adv_ae.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                            a[3][i].set_xticks(())
                                            a[3][i].set_yticks(())
                                            a[3][i].set_title("Adv Images Decoded" + "[" + str(i + 1) + "]")
                                        plt.suptitle("Epoch:" + str(epoch + 1))
                                        plt.draw()
                                        plt.pause(0.05)
                                        if (epoch + 1) % 5 == 0:
                                            img = plt.gcf()
                                            img.savefig(args.image_path + str(epoch + 1) + ".png")
                                        plt.show()

                                    batch_id_test += 1

                                # print(correct_clean_cla, total_test)

                                print("Clean Test Set Classification Accuracy：%.2f%% | Adv Test Set Classification Accuracy：%.2f%%"
                                      % (correct_clean_cla / total_test * 100, correct_adv_cla / total_test * 100))
                                acc_clean = correct_clean_cla / total_test * 100
                                acc_adv = correct_adv_cla / total_test * 100
                                # 保存测试集准确率至acc.txt文件中
                                f2.write("Epoch = %03d, Clean Set Accuracy = %.2f%%, Adv Set Accuracy = %.2f%%" %
                                         (epoch + 1, acc_clean, acc_adv))
                                f2.write("\n")
                                f2.flush()

                                print("Clean Test Set AutoEncoder Loss：%.4f | Adv Test Set AutoEncoder Loss：%.4f" %
                                      (sum_loss_clean, sum_loss_ae))
                                # 保存测试集loss至loss.txt文件中
                                f4.write("Epoch = %03d, Clean Set Loss = %.4f, Adv Set Loss = %.4f" %
                                         (epoch + 1, sum_loss_clean, sum_loss_ae))
                                f4.write("\n")
                                f4.flush()

                                print("Saving model!")
                                torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch + 1))
                                print("Model saved!")

                                # 记录最小adv测试集acc并写入best_acc.txt文件中
                                if acc_adv > best_acc_test_adv:
                                    # if epoch != 50:
                                    #     os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                                    best_acc_test_adv = acc_adv
                                    f5 = open(args.best_acc_file_path, "w")
                                    f5.write("Epoch = %d, Best Acc of Adv Set = %.4f" % (epoch + 1, best_acc_test_adv))
                                    f5.close()
                                    best_epoch = epoch + 1

                                    # print("Saving model!")
                                    # torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch + 1))
                                    # print("Model saved!")

                                # 记录最小测试集loss并写入min_loss.txt文件中
                                if sum_loss_adv < min_loss_adv:
                                    min_loss_adv = sum_loss_adv
                                    f6 = open(args.min_loss_file_path, "w")
                                    f6.write("Epoch = %d, Min Loss = %.4f" % (epoch + 1, sum_loss_adv))
                                    f6.close()

if __name__ == "__main__":
    best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_" \
                          + str(282) + ".pth"
    best_ae_model_path = "D:/python_workplace/resnet-AE/checkpoint/Autoencoder/cifar10/block_4/model/model_" \
                         + str(250) + ".pth"
    # best_ae_model_path = "D:/python_workplace/resnet-AE/checkpoint/Autoencoder/cifar10/block_3/model/model_" \
    #                       + str(130) + ".pth"
    # best_ae_model_path = "D:/python_workplace/resnet-AE/checkpoint/Autoencoder/cifar10/block_2/model/model_" \
    #                      + str(190) + ".pth"
    joint_train(best_cla_model_path, best_ae_model_path, "cuda:1")
