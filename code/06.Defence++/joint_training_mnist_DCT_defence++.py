import cv2
import numpy as np
import pickle
import time
import argparse
import os
import torch
import random
import torchvision
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.misc import imresize
from torchsummary import summary
from models.joint_training_lp_hc_mnist import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder
from torch.utils.data import DataLoader


torch.cuda.manual_seed(10)
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


def load_images(input_dir, batch_shape_clean, batch_shape_adv):
    image_clean = np.zeros(batch_shape_clean)
    image_adv = np.zeros(batch_shape_adv)
    gt_4 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 14, 14])
    gt_3 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 7, 7])
    gt_2 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 4, 4])
    gt_1 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 2, 2])
    h_4 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 14, 14])
    labels = np.zeros(batch_shape_clean[0])
    idx = 0
    batch_size = batch_shape_adv[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_clean_list = data_dict["images_clean"]
    images_adv_list = data_dict["images_adv"]
    labels_list = data_dict["labels"]

    for j in range(len(images_clean_list)):
        image_clean[idx, 0, :, :] = images_clean_list[j]
        image_adv[idx, 0, :, :] = images_adv_list[j]
        image_clean_j = np.array(images_clean_list[j])
        dct_gt = np.zeros([28, 28, 1])
        dct_gt[:, :, 0] = image_clean_j[0, :, :]
        # cv2.imshow("gray", dct_gt)
        dct_gt = cv2.dct(dct_gt.astype(np.float32))
        # cv2.imshow("dct", dct_gt)
        idct_gt = cv2.idct(dct_gt)
        # cv2.imshow("idct", idct_gt)
        gt_4_j = cv2.idct(dct_gt[0:14, 0:14])
        # cv2.imshow("idct", gt_4_j)
        # cv2.waitKey(0)
        gt_3_j = cv2.idct(dct_gt[0:7, 0:7])
        gt_2_j = cv2.idct(dct_gt[0:4, 0:4])
        gt_1_j = cv2.idct(dct_gt[0:2, 0:2])
        h_4_j = cv2.idct(dct_gt[14:28, 14:28])
        gt_4[idx, 0, :, :] = gt_4_j
        gt_3[idx, 0, :, :] = gt_3_j
        gt_2[idx, 0, :, :] = gt_2_j
        gt_1[idx, 0, :, :] = gt_1_j
        h_4[idx, 0, :, :] = h_4_j

        labels[idx] = labels_list[j]

        idx += 1
        if idx == batch_size:
            # yield idx, gt_1, gt_2, gt_3, gt_4, h_4, image_clean, image_adv, labels
            yield idx, gt_1, gt_2, gt_3, gt_4, image_clean, image_adv, labels
            image_clean = np.zeros(batch_shape_clean)
            image_adv = np.zeros(batch_shape_adv)
            gt_4 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 14, 14])
            gt_3 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 7, 7])
            gt_2 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 4, 4])
            gt_1 = np.zeros([batch_shape_clean[0], batch_shape_clean[1], 2, 2])
            labels = np.zeros(batch_shape_clean[0])
            idx = 0

    if idx > 0:
        # yield idx, gt_1, gt_2, gt_3, gt_4, h_4, image_clean, image_adv, labels
        yield idx, gt_1, gt_2, gt_3, gt_4, image_clean, image_adv, labels


def joint_train(best_cla_model_path, block, device_used):
    # Device configuration
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument("--input_path", type=str, default="D:/ETUDE/inputData/mnist/",
                        help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--image_size", type=int, default=28, help="Size of each input images.")
    parser.add_argument("--epochs", type=int, default=250, help="Epoch default:50.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.0001")
    parser.add_argument("--input_dir_trainSet", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/train/train.pkl",
                        help="data set dir path")
    parser.add_argument("--input_dir_testSet", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/test/test.pkl",
                        help="data set dir path")
    parser.add_argument("--model_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/model/",
                        help="Save model path")
    parser.add_argument("--image_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/image/",
                        help="Save log file")
    parser.add_argument("--acc_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new_new/mnist/block_"
                                + str(block) + "/acc.txt",
                        help="Save accuracy file")
    parser.add_argument("--best_acc_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/best_acc.txt",
                        help="Save best accuracy file")
    parser.add_argument("--log_cla_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/log_cla.txt",
                        help="Save log file")
    parser.add_argument("--log_ae_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/log_ae.txt",
                        help="Save log file")
    parser.add_argument("--loss_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/loss.txt",
                        help="Save log file")
    parser.add_argument("--min_loss_file_path", type=str,
                        default="D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_"
                                + str(block) + "/min_loss.txt",
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

    model_1 = joint_model_1(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_2 = joint_model_2(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_3 = joint_model_3(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_4 = joint_model_4(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model_list = [model_1, model_2, model_3, model_4]
    model = model_list[block - 1]

    pretrained_dict = torch.load(best_cla_model_path)
    model_dict = model.state_dict()

    pretrained_weight_key = []
    pretrained_weight_value = []
    model_weight_key = []
    model_weight_value = []
    for k, v in pretrained_dict.items():
        # print(k)
        pretrained_weight_key.append(k)
        pretrained_weight_value.append(v)

    # pretrained_weight_key = pretrained_weight_key[1:]
    # pretrained_weight_value = pretrained_weight_value[1:]
    # print(pretrained_weight_key)
    for k, v in model_dict.items():
        # print(k)
        model_weight_key.append(k)
        model_weight_value.append(v)
    # print(model_weight_key)
    new_dict = {}
    for i in range(len(model_dict)):
        if i < len(pretrained_weight_key):
            new_dict[model_weight_key[i]] = pretrained_weight_value[i]
        else:
            new_dict[model_weight_key[i]] = model_weight_value[i]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    # summary(model, (1, 28, 28))

    batchShape_clean = [args.batch_size, 1, args.image_size, args.image_size]
    batchShape_adv = [args.batch_size, 1, args.image_size, args.image_size]

    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    # criterion
    criterion_cla = nn.CrossEntropyLoss().to(device)
    criterion_ae = nn.MSELoss()

    length = len(train_loader)

    # initialize figure
    f, a = plt.subplots(6, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    best_acc_test_adv = 0
    min_loss_clean = 100
    min_loss_adv = 100
    best_epoch = 1
    print("Start Joint Training!")
    with open(args.log_cla_file_path, "w") as f1:
        with open(args.acc_file_path, "w") as f2:
            with open(args.log_ae_file_path, "w") as f3:
                with open(args.loss_file_path, "w") as f4:
                    for epoch in range(1, args.epochs + 1):
                        if epoch <= 100:
                            args.lr = 0.01
                        elif epoch > 100 & epoch <= 200:
                            args.lr = 0.001
                        else:
                            args.lr = 0.0001

                        # Optimization
                        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

                        # training
                        print("Training Epoch: %d" % epoch)
                        sum_loss_cla = 0.0
                        sum_loss_ae = 0.0
                        correct_cla = 0.0
                        total_train = 0
                        batch = 1
                        for batchSize, gt_1, gt_2, gt_3, gt_4, images_clean, images_adv, labels in load_images(
                                args.input_dir_trainSet,
                                batchShape_clean,
                                batchShape_adv):
                            start = time.time()
                            # data prepare
                            inputs_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
                            gt_1 = torch.from_numpy(gt_1).type(torch.FloatTensor).to(device)
                            gt_2 = torch.from_numpy(gt_2).type(torch.FloatTensor).to(device)
                            gt_3 = torch.from_numpy(gt_3).type(torch.FloatTensor).to(device)
                            gt_4 = torch.from_numpy(gt_4).type(torch.FloatTensor).to(device)

                            model.to(device)
                            model.train()
                            optimizer.zero_grad()

                            # forward + backward
                            outputs_cla, de_block_1, de_block_2, de_block_3, de_block_4, outputs_ae = model(
                                inputs_clean)
                            loss_cla = criterion_cla(outputs_cla, labels)
                            loss_block_1 = criterion_ae(de_block_1, gt_1)
                            loss_block_2 = criterion_ae(de_block_2, gt_2)
                            loss_block_3 = criterion_ae(de_block_3, gt_3)
                            loss_block_4 = criterion_ae(de_block_4, gt_4)
                            loss_ae = criterion_ae(outputs_ae, inputs_clean)
                            # loss = loss_cla + 0.8 * loss_ae + 0.1 * (loss_block_1 + loss_block_2 + loss_block_3 + loss_block_4)
                            loss = loss_cla + 0.8 * loss_ae + 0.01 * (loss_block_3 + loss_block_4)
                            # loss = loss_cla + 0.8 * loss_ae
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
                            print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs"
                                  % (epoch, args.epochs, batch, length, sum_loss_cla / batch,
                                     correct_cla / total_train * 100, args.lr, (end - start)))
                            f1.write(
                                "[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.4f | Time: %.3fs"
                                % (epoch, args.epochs, batch, length, sum_loss_cla / batch,
                                   correct_cla / total_train * 100, args.lr, (end - start)))
                            f1.write("\n")
                            f1.flush()

                            print("Log of AutoEncoder:")
                            print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.04f | Time: %.03fs"
                                  % (epoch, args.epochs, batch, length, sum_loss_ae / batch,
                                     args.lr, (end - start)))
                            f3.write("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs"
                                     % (epoch, args.epochs, batch, length, sum_loss_ae / batch,
                                        args.lr, (end - start)))
                            f3.write("\n")
                            f3.flush()

                            batch += 1

                        # testing
                        if epoch % 1 == 0:
                            print("Waiting for Testing Cleaning Set!")
                            with torch.no_grad():
                                correct_clean_cla = 0
                                correct_adv_cla = 0
                                total_test = 0
                                sum_loss_clean = 0
                                sum_loss_adv = 0
                                batch_id_test = 1
                                for batchSize, gt_1, gt_2, gt_3, gt_4, images_clean, images_adv, labels in load_images(
                                        args.input_dir_testSet,
                                        batchShape_clean,
                                        batchShape_adv):
                                    model.eval()
                                    images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                                    images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                                    # gt_1 = torch.from_numpy(gt_1).type(torch.FloatTensor).to(device)
                                    # gt_2 = torch.from_numpy(gt_2).type(torch.FloatTensor).to(device)
                                    gt_3 = torch.from_numpy(gt_3).type(torch.FloatTensor).to(device)
                                    gt_4 = torch.from_numpy(gt_4).type(torch.FloatTensor).to(device)
                                    labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
                                    model.to(device)
                                    total_test += batchSize

                                    # 测试clean数据集的测试集
                                    outputs_clean_cla, de_block_1, de_block_2, de_block_3, de_block_4, outputs_clean_ae = model(
                                        images_clean)
                                    _, predicted_clean_cla = torch.max(outputs_clean_cla.data, 1)
                                    correct_clean_cla += (predicted_clean_cla == labels).sum().item()

                                    loss_clean = criterion_ae(outputs_clean_ae, images_clean)
                                    sum_loss_clean += loss_clean.cpu().data.numpy()

                                    # 测试adv数据集的测试集
                                    outputs_adv_cla, de_block_1, de_block_2, de_block_3, de_block_4, outputs_adv_ae = model(
                                        images_adv)
                                    _, predicted_adv_cla = torch.max(outputs_adv_cla.data, 1)
                                    correct_adv_cla += (predicted_adv_cla == labels).sum().item()

                                    loss_adv = criterion_ae(outputs_adv_ae, images_clean)
                                    sum_loss_adv += loss_adv.cpu().data.numpy()

                                    if batch_id_test == 4:
                                        # first row:clean images
                                        for i in range(5):
                                            a[0][i].clear()
                                            a[0][i].imshow(
                                                np.reshape(images_clean.cpu().data.numpy()[i, 0, :, :], (28, 28)),
                                                cmap="gray")
                                            a[0][i].set_xticks(())
                                            a[0][i].set_yticks(())
                                            a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")

                                        # second row:clean images(GCT trans)
                                        for i in range(5):
                                            a[1][i].clear()
                                            a[1][i].imshow(np.reshape(gt_4.cpu().data.numpy()[i, 0, :, :], (14, 14)),
                                                           cmap="gray")
                                            a[1][i].set_xticks(())
                                            a[1][i].set_yticks(())
                                            a[1][i].set_title("GCT Images(14*14)" + "[" + str(i + 1) + "]")

                                        # third row:clean images(GCT trans)
                                        for i in range(5):
                                            a[2][i].clear()
                                            a[2][i].imshow(np.reshape(gt_3.cpu().data.numpy()[i, 0, :, :], (7, 7)),
                                                           cmap="gray")
                                            a[2][i].set_xticks(())
                                            a[2][i].set_yticks(())
                                            a[2][i].set_title("GCT Images(7*7)" + "[" + str(i + 1) + "]")

                                        # fourth row:adv images
                                        for i in range(5):
                                            a[3][i].clear()
                                            a[3][i].imshow(
                                                np.reshape(images_adv.cpu().data.numpy()[i, 0, :, :], (28, 28)),
                                                cmap="gray")
                                            a[3][i].set_xticks(())
                                            a[3][i].set_yticks(())
                                            a[3][i].set_title("Adv Images" + "[" + str(i + 1) + "]")

                                        # fifth row:decoded clean images
                                        for i in range(5):
                                            a[4][i].clear()
                                            a[4][i].imshow(np.reshape(outputs_clean_ae.cpu().data.numpy()[i], (28, 28)),
                                                           cmap="gray")
                                            a[4][i].set_xticks(())
                                            a[4][i].set_yticks(())
                                            a[4][i].set_title("Clean Images Decoded" + "[" + str(i + 1) + "]")

                                        # sixth row:decoded adv images
                                        for i in range(5):
                                            a[5][i].clear()
                                            a[5][i].imshow(np.reshape(outputs_adv_ae.cpu().data.numpy()[i], (28, 28)),
                                                           cmap="gray")
                                            a[5][i].set_xticks(())
                                            a[5][i].set_yticks(())
                                            a[5][i].set_title("Adv Images Decoded" + "[" + str(i + 1) + "]")
                                        plt.suptitle("Epoch:" + str(epoch))
                                        plt.draw()
                                        plt.pause(0.05)
                                        if (epoch) % 5 == 0:
                                            img = plt.gcf()
                                            img.savefig(args.image_path + str(epoch) + ".png")
                                        plt.show()

                                    batch_id_test += 1

                                print(
                                    "Clean Test Set Classification Accuracy：%.2f%% | Adv Test Set Classification Accuracy：%.2f%%"
                                    % (correct_clean_cla / total_test * 100, correct_adv_cla / total_test * 100))
                                acc_clean = correct_clean_cla / total_test * 100
                                acc_adv = correct_adv_cla / total_test * 100
                                # 保存测试集准确率至acc.txt文件中
                                f2.write("Epoch = %03d, Clean Set Accuracy = %.2f%%, Adv Set Accuracy = %.2f%%" %
                                         (epoch, acc_clean, acc_adv))
                                f2.write("\n")
                                f2.flush()

                                print("Clean Test Set AutoEncoder Loss：%.4f | Adv Test Set AutoEncoder Loss：%.4f" %
                                      (sum_loss_clean, sum_loss_ae))
                                # 保存测试集loss至loss.txt文件中
                                f4.write("Epoch = %03d, Clean Set Loss = %.4f, Adv Set Loss = %.4f" %
                                         (epoch, sum_loss_clean, sum_loss_ae))
                                f4.write("\n")
                                f4.flush()

                                # 记录最小adv测试集acc并写入best_acc.txt文件中
                                if acc_adv > best_acc_test_adv:
                                    if epoch != 1:
                                        os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                                    best_acc_test_adv = acc_adv
                                    f5 = open(args.best_acc_file_path, "w")
                                    f5.write("Epoch = %d, Best Acc of Adv Set = %.4f" % (epoch, best_acc_test_adv))
                                    f5.close()
                                    best_epoch = epoch

                                    print("Saving model!")
                                    torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch))
                                    print("Model saved!")

                                # 记录最小测试集loss并写入min_loss.txt文件中
                                if sum_loss_adv < min_loss_adv:
                                    min_loss_adv = sum_loss_adv
                                    f6 = open(args.min_loss_file_path, "w")
                                    f6.write("Epoch = %d, Min Loss = %.4f" % (epoch, sum_loss_adv))
                                    f6.close()

if __name__ == "__main__":
    best_model_path = "D:/ETUDE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18_new/mnist/block_1/model/model_5.pth"
    joint_train(best_model_path, 1, "cuda:0")

    # a = [[1, 3], [2, 3]]
    # print(a)

    # from scipy.misc import imsave
    #
    # a = [[-5.9022129e-01, -2.1404386e-03, -2.7523806e-02, -1.3553943e-01, 2.7137798e-01, -7.9754509e-02,
    #       -1.1399074e-01, -3.4354416e-01, -9.4718289e-01, -2.0922692e-01, -1.1398347e-01,  7.8791358e-02,
    #       1.0739907e-02, 1.1887258e-01],
    #      [2.2531961e-01, 4.0612796e-01,  3.2301143e-01,  3.1732079e-01, -9.0400554e-02, -8.4624118e-01,
    #       -8.6783129e-01, -4.9416642e+00, -8.1544333e+00, -7.3293328e-01, -1.6965272e-01, 1.2215569e-01,
    #       -7.5265586e-02, -2.6907040e-02],
    #      [2.7019030e-01, -3.3660924e-01, -8.3026731e-01, -5.4135580e-02, -1.0768517e-02, -6.1144447e-01,
    #        -2.3190396e+00, -6.8822203e+00, -7.5577092e+00, -8.1835335e-01, -4.6262121e-01, -6.7844421e-01,
    #        -5.0596333e-01, 1.7537659e-01],
    #      [-5.5666361e-02, -4.0659559e-01, -2.0269533e-01, -3.2714847e-01, -8.6876976e-01, -4.8597619e-01,
    #       -3.2473536e+00, -8.6322231e+00, -3.2684660e+00, -1.4915757e-01, -7.1393442e-01, -3.9364737e-01,
    #       1.8034668e-01, 7.6201223e-02],
    #      [-1.1534001e-01, -1.6492519e-01, 3.7254840e-01, -2.1406291e-01, -1.1488141e+00, -3.1368089e+00,
    #       -7.5660701e+00, -5.0078793e+00, -1.5921327e+00, -1.5644690e+00, -1.2045854e+00, -2.3327851e-01,
    #       -5.0543183e-01, -1.9574545e-01],
    #      [-2.5801018e-01, -1.8775530e-01, -2.9296178e-01, -1.9421053e-01, -1.1279302e+00, -7.6172700e+00,
    #       -3.2536540e+00, -1.2867738e+00, -1.4660386e+00, -1.3610239e+00, -6.8820280e-01, -9.5248359e-01,
    #       -6.1340612e-01, 2.2957014e-01],
    #      [2.7082853e-02, 2.6754689e-01, 3.9508319e-01, 2.1488754e-01, -1.6122983e+00, -8.1173115e+00,
    #        -2.1989005e+00, -4.3497348e+00, -5.6937165e+00, -6.3333068e+00, -5.6227646e+00, -3.4104452e+00,
    #        -5.8587039e-01, 2.9276040e-01],
    #      [-1.4684308e-01, -2.8155394e-02, -6.4440125e-01, -6.4525998e-01, -4.9671869e+00, -8.3466463e+00,
    #       -7.6859741e+00, -3.7512043e+00, -1.2315794e+00, -3.0445611e+00, -9.2639923e+00, -5.0685244e+00,
    #       -3.5685498e-01, 1.6196784e-02],
    #      [-4.5663536e-01, -3.0439460e-01, -9.5909011e-01, -4.3113656e+00, -7.5858779e+00, -6.3936152e+00,
    #       -1.3250725e+00, -2.6811832e-01, -2.8221646e-01, -1.5355681e+00, -8.8651686e+00, -3.7889714e+00,
    #       2.9019731e-01, 2.3041019e-01],
    #      [-6.3166045e-02, -9.4167108e-04, -8.4223217e-01, -5.4247279e+00, -9.2277012e+00, -5.6644859e+00,
    #       -4.0852404e-01, -3.2915571e-01, -1.2978580e+00, -5.8466396e+00, -7.8959904e+00, -2.9457111e+00,
    #       -7.9862103e-02, -3.5519913e-02],
    #      [5.6028273e-02, 3.1534076e-01, -5.9184468e-01, -5.8184619e+00, -8.7187309e+00, -6.9163909e+00,
    #       -5.9178782e+00, -6.6416435e+00, -7.3170891e+00, -5.6888995e+00, -3.2720938e+00, -5.9985942e-01,
    #       -6.6075951e-02, -8.3748996e-02],
    #      [3.6995617e-01, 4.1954032e-01, -1.8319324e-01, -5.6445581e-01, -6.6161853e-01, -6.8130344e-01,
    #        -1.0742129e+00, -1.9216149e+00, -1.4673827e+00, -4.5973313e-01, -6.8932730e-01, -4.2329934e-01,
    #        -3.4630665e-01,  2.0048775e-02],
    #      [-1.2589937e-01, -2.9631150e-01, -3.6512040e-02, -2.2850581e-01, 1.6326930e-01, -1.1042043e-01,
    #       -1.1035203e+00, -4.3534064e-01, -4.7984593e-02, -1.8832810e-01, -3.7169835e-01, -2.7886498e-01,
    #       3.2461375e-01, 2.5135803e-01],
    #      [7.2081690e-03, -1.4073193e-01, -2.2120443e-01, 1.5738921e-02, -1.4758061e-01, -2.1714355e-01,
    #        -3.1891599e-01, 1.6056387e-01, -9.5706090e-02, -5.2479742e-04, 3.3304247e-01, 4.2923933e-01,
    #        2.9196942e-01, 2.8166005e-01]]
    #
    # b = [[-2.72150934e-01, -2.80536741e-01, -1.90925106e-01, -1.96808115e-01, -1.24246821e-01, -1.28075257e-01,
    #       3.90000902e-02, 4.02018055e-02, 1.19148090e-01, 1.22819416e-01, -1.82188377e-01, -1.87802166e-01,
    #       -1.06082618e-01, -1.09351359e-01],
    #      [-2.79463083e-01, -2.68178344e-01, -1.96054906e-01, -1.88138202e-01, -1.27585098e-01, -1.22433200e-01,
    #       4.00479473e-02, 3.84308100e-02, 1.22349374e-01, 1.17408901e-01, -1.87083423e-01, -1.79528981e-01,
    #       -1.08932860e-01, -1.04534142e-01],
    #      [-2.83353090e-01, -2.92084098e-01, -8.91117156e-02, -9.18575302e-02, 2.47178320e-02, 2.54794657e-02,
    #       6.42941296e-01, 6.62752330e-01, 4.42038298e-01, 4.55658883e-01, 1.52744632e-02, 1.57451183e-02,
    #       -3.32716823e-01, -3.42968881e-01],
    #      [-2.90966243e-01, -2.79217005e-01, -9.15059745e-02, -8.78109634e-02, 2.53819525e-02, 2.43570283e-02,
    #       6.60215914e-01, 6.33556366e-01, 4.53915030e-01, 4.35585886e-01, 1.56848598e-02, 1.50515037e-02,
    #       -3.41656297e-01, -3.27860177e-01],
    #      [-4.52921614e-02, -4.66877557e-02, -9.88407433e-02, -1.01886332e-01, 4.74283576e-01, 4.88897771e-01,
    #       8.93132806e-01, 9.20653045e-01, 3.25631313e-02, 3.35665047e-02, 6.76316209e-04, 6.97155658e-04,
    #       -3.82104486e-01, -3.93878341e-01],
    #      [-4.65090759e-02, -4.46310379e-02, -1.01496406e-01, -9.73979682e-02, 4.87026691e-01, 4.67360526e-01,
    #       9.17129576e-01, 8.80095840e-01, 3.34380418e-02, 3.20878103e-02, 6.94487593e-04, 6.66444132e-04,
    #       -3.92370909e-01, -3.76526952e-01],
    #      [-1.70647278e-01, -1.75905466e-01, -2.30442192e-02, -2.37542838e-02, 6.76578522e-01, 6.97426021e-01,
    #       7.60759711e-01, 7.84201145e-01, 4.12420571e-01, 4.25128549e-01, 8.39699745e-01, 8.65573525e-01,
    #       -8.27111676e-02, -8.52597654e-02],
    #      [-1.75232247e-01, -1.68156356e-01, -2.36633737e-02, -2.27078442e-02, 6.94756866e-01, 6.66702569e-01,
    #       7.81199872e-01, 7.49655008e-01, 4.23501521e-01, 4.06400502e-01, 8.62260878e-01, 8.27442765e-01,
    #       -8.49334598e-02, -8.15038458e-02],
    #      [-1.34057686e-01, -1.38188437e-01, 4.32815105e-01, 4.46151495e-01, 8.41880739e-01, 8.67821753e-01,
    #       1.33766547e-01, 1.37888312e-01, 5.64733803e-01, 5.82135022e-01, 9.53738332e-01, 9.83126044e-01,
    #       -1.20488778e-01, -1.24201417e-01],
    #      [-1.37659565e-01, -1.32100865e-01, 4.44444031e-01, 4.26497340e-01, 8.64500463e-01, 8.29591870e-01,
    #       1.37360588e-01, 1.31813958e-01, 5.79907119e-01, 5.56490421e-01, 9.79363501e-01, 9.39816713e-01,
    #       -1.23726077e-01, -1.18730016e-01],
    #      [-1.60916343e-01, -1.65874675e-01, 3.07145268e-01, 3.16609383e-01, 9.45298374e-01, 9.74426031e-01,
    #       3.76825422e-01, 3.88436586e-01, 5.58470666e-01, 5.75678885e-01, 2.87362635e-01, 2.96217203e-01,
    #       -4.29525971e-01, -4.42761004e-01],
    #      [-1.65239856e-01, -1.58567458e-01, 3.15397680e-01, 3.02661896e-01, 9.70696747e-01, 9.31499958e-01,
    #       3.86950016e-01, 3.71324927e-01, 5.73475718e-01, 5.50318718e-01, 2.95083523e-01, 2.83168048e-01,
    #       -4.41066504e-01, -4.23256218e-01],
    #      [-3.14083397e-01, -3.23761284e-01, -1.08177640e-01, -1.11510932e-01, -4.61373758e-03, -4.75590117e-03,
    #       -2.39978656e-01, -2.47373149e-01, -9.87528712e-02, -1.01795755e-01, -2.73478657e-01, -2.81905413e-01,
    #       -2.15868473e-01, -2.22520053e-01],
    #      [-3.22522223e-01, -3.09498757e-01, -1.11084171e-01, -1.06598578e-01, -4.73769987e-03, -4.54639085e-03,
    #       -2.46426433e-01, -2.36475706e-01, -1.01406172e-01, -9.73113850e-02, -2.80826509e-01, -2.69486725e-01,
    #       -2.21668452e-01, -2.12717459e-01]]
    #
    # # plt.imshow(np.array(a), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # # plt.imshow(np.array(b), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # imsave("D:/picture/Figure-5/高频原始.png", a)
    # imsave("D:/picture/Figure-5/低频原始.png", b)
    #
    # c = [[0 for i in range(14)] for i in range(14)]
    #
    # for i in range(14):
    #     for j in range(14):
    #         c[i][j] = a[i][j] + b[i][j]
    #
    # # plt.imshow(np.array(c), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # imsave("D:/picture/Figure-5/相加原始.png", c)
    #
    # min_num, max_num = 1, 0
    # # for i in range(14):
    # #     for j in range(14):
    # #         if abs(a[i][j]) > max_num:
    # #             max_num = abs(a[i][j])
    # #         if abs(a[i][j]) < min_num:
    # #             min_num = abs(a[i][j])
    # #
    # # print(min_num, max_num)
    #
    # for i in range(14):
    #     for j in range(14):
    #         if abs(b[i][j]) > max_num:
    #             max_num = abs(b[i][j])
    #             # print(i, j)
    #         if abs(b[i][j]) < min_num:
    #             min_num = abs(b[i][j])
    #             # print(i, j)
    #
    # print(min_num, max_num)
    #
    # for i in range(14):
    #     for j in range(14):
    #         a[i][j] = (max_num - abs(a[i][j])) / (max_num - min_num)
    #
    # for i in range(14):
    #     for j in range(14):
    #         b[i][j] = (max_num - abs(b[i][j])) / (max_num - min_num)
    #
    # # print(a, b)
    #
    # # plt.imshow(np.array(a), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # # plt.imshow(np.array(b), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # imsave("D:/picture/Figure-5/高频new.png", a)
    # imsave("D:/picture/Figure-5/低频new.png", b)
    #
    # for i in range(14):
    #     for j in range(14):
    #         c[i][j] = a[i][j] + b[i][j]
    #
    # # plt.imshow(np.array(c), cmap="gray")
    # # plt.axis('off')
    # # plt.show()
    # imsave("D:/picture/Figure-5/相加new.png", c)