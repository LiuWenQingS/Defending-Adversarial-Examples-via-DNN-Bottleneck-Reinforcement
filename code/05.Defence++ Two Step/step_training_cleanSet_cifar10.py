import torch
from torch import nn
import torchvision
from torchsummary import summary
import argparse
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from AutoEncoder_resnet_cifar10 import ResNet_AutoEncoder, BasicBlock_decoder, BasicBlock_encoder


def train_ae(best_cla_model_path, round, block, ae_training_set, device_used):
    # Device configuration
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    # parser.add_argument("--input_path", type=str, default="D:/python_workplace/resnet-AE/inputData/cifar/cifar10/cifar-10-batches-py/",
    #                     help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--input_path", type=str,
                        default="C:/Users/WenqingLiu/cifar/cifar10/cifar-10-batches-py/",
                        help="data set dir path")
    parser.add_argument("--image_size", type=int, default=32, help="Size of each input images.")
    parser.add_argument("--batch_size", type=int, default=256, help="How many images process at one time.")
    parser.add_argument("--epochs", type=int, default=100, help="How many times for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--model_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) + "/model/",
                        help="Save model path")
    parser.add_argument("--min_loss_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) + "/min_loss.txt",
                        help="Save min loss")
    parser.add_argument("--image_compare", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) + "/images/",
                        help="Save images")
    parser.add_argument("--log_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) + "/log.txt",
                        help="Save log file")
    parser.add_argument("--loss_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/cifar10/block_" + str(block) + "/round_" + str(round) + "/loss.txt",
                        help="Save loss file")

    args = parser.parse_args()

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
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

    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    model = ResNet_AutoEncoder(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
    model.to(device)
    # summary(model, (3, 32, 32))
    # print(model)

    # # Load pre-trained weights
    # pretrained_dict = torch.load(args.checkpoint_path_ae)
    # model_dict = model.state_dict()
    # new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # model.to(device)

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

    pretrained_weight_key = pretrained_weight_key[0:-2]
    pretrained_weight_value = pretrained_weight_value[0:-2]
    # print(pretrained_weight_key)
    for k, v in model_dict.items():
        # print(k)
        model_weight_key.append(k)
        model_weight_value.append(v)
    # print(model_weight_key)
    new_dict = {}
    for i in range(6 + 4 * 6 * block + 6 * (block - 1)):
        new_dict[model_weight_key[i]] = pretrained_weight_value[i]

    # for k, v in new_dict.items():
    #     print(k)

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # criterion
    criterion = nn.MSELoss()

    # initialize figure
    f, a = plt.subplots(2, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    min_loss = 10000
    best_epoch = 1
    print("Start Training AutoEncoder!")
    with open(args.log_file_path, "w") as f1:
        with open(args.loss_file_path, "w") as f2:
            for epoch in range(0, args.epochs):
                if epoch + 1 <= 50:
                    args.lr = 0.01
                elif epoch + 1 > 50 & epoch + 1 <= 80:
                    args.lr = 0.001
                else:
                    args.lr = 0.0001

                # Optimization
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

                print("Epoch: %d" % (epoch + 1))
                sum_loss = 0.0
                batch_id = 1
                for i, data in enumerate(train_loader, 0):
                    start = time.time()

                    # model prepare
                    model.train()
                    optimizer.zero_grad()

                    # data prepare
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward + backward
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.cpu().data.numpy()

                    end = time.time()

                    print("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.04f | Time: %.03fs"
                          % (
                          epoch + 1, args.epochs, batch_id, (50000 / args.batch_size) + 1, sum_loss / batch_id, args.lr,
                          (end - start)))
                    f1.write("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs"
                             % (epoch + 1, args.epochs, batch_id, (50000 / args.batch_size) + 1, sum_loss / batch_id,
                                args.lr, (end - start)))
                    f1.write("\n")
                    f1.flush()
                    batch_id += 1

                # 每训练完一个epoch测试loss
                print("Waiting for Testing!")
                with torch.no_grad():
                    sum_loss_test = 0
                    batch_id_test = 1
                    for i, data in enumerate(test_loader, 0):
                        model.eval()
                        inputs, labels = data
                        # print(inputs[0])
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                        sum_loss_test += loss.cpu().data.numpy()
                        outputs = torch.clamp(outputs, 0, 1)
                        # print(outputs[0])

                        if batch_id_test == 4:
                            # 观察原图以及去噪后的图片
                            # first row:clean images
                            for i in range(5):
                                a[0][i].clear()
                                a[0][i].imshow(inputs.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                a[0][i].set_xticks(())
                                a[0][i].set_yticks(())
                                a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")

                            # second row:decoded images
                            for i in range(5):
                                a[1][i].clear()
                                a[1][i].imshow(outputs.cpu().data.numpy()[i, :, :, :].transpose((1, 2, 0)))
                                a[1][i].set_xticks(())
                                a[1][i].set_yticks(())
                                a[1][i].set_title("Decoded Images" + "[" + str(i + 1) + "]")
                            plt.suptitle("Epoch:" + str(epoch + 1))
                            plt.draw()
                            plt.pause(0.05)
                            if (epoch + 1) % 1 == 0:
                                img = plt.gcf()
                                img.savefig(args.image_compare + str(epoch + 1) + ".png")
                            plt.show()

                        batch_id_test += 1
                    print("Test Set Loss：%.4f" % sum_loss_test)

                    # 保存测试集loss至loss.txt文件中
                    f2.write("Epoch=%03d,Loss= %.4f" % (epoch + 1, sum_loss_test))
                    f2.write("\n")
                    f2.flush()
                    # 记录最小测试集loss并写入min_loss.txt文件中
                    if sum_loss_test < min_loss:
                        if epoch != 0:
                           os.remove(args.model_path + "model_" + str(best_epoch) + ".pth")
                        min_loss = sum_loss_test
                        f3 = open(args.min_loss_path, "w")
                        f3.write("Epoch=%d,min_loss= %.4f" % (epoch + 1, sum_loss_test))
                        f3.close()
                        best_epoch = epoch + 1

                        print("Saving model!")
                        torch.save(model.state_dict(), "%s/model_%d.pth" % (args.model_path, epoch + 1))
                        print("Model saved!")
    return best_epoch


if __name__ == "__main__":
    train_ae("D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_282.pth", 1, 1,
             "AE_CleanSet", "cuda:0")


