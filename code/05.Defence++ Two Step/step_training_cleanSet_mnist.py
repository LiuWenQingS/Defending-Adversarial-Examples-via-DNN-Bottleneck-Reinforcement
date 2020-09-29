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
from AutoEncoder_resnet_mnist import ResNet_AutoEncoder, BasicBlock_decoder, BasicBlock_encoder


def train_ae(best_cla_model_path, best_ae_model_path, round, block, ae_training_set, device_used):
    # Device configuration
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument("--input_path", type=str, default="D:/python_workplace/resnet-AE/inputData/mnist/",
                        help="image dir path default: ../inputData/mnist/.")
    parser.add_argument("--image_size", type=int, default=28, help="Size of each input images.")
    parser.add_argument("--batch_size", type=int, default=512, help="How many images process at one time.")
    parser.add_argument("--epochs", type=int, default=100, help="How many times for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--model_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/mnist/block_" + str(block) + "/round_" + str(round) + "/model/",
                        help="Save model path")
    parser.add_argument("--min_loss_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/mnist/block_" + str(block) + "/round_" + str(round) + "/min_loss.txt",
                        help="Save min loss")
    parser.add_argument("--image_compare", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/mnist/block_" + str(block) + "/round_" + str(round) + "/images/",
                        help="Save images")
    parser.add_argument("--log_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/mnist/block_" + str(block) + "/round_" + str(round) + "/log.txt",
                        help="Save log file")
    parser.add_argument("--loss_file_path", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/" + ae_training_set +
                                "/Autoencoder/ResNet18/mnist/block_" + str(block) + "/round_" + str(round) + "/loss.txt",
                        help="Save loss file")

    args = parser.parse_args()

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
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

    model = ResNet_AutoEncoder(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2]).to(device)
    # summary(model, (1, 28, 28))

    # Load pre-trained weights
    pretrained_cla_dict = torch.load(best_cla_model_path)
    model_dict = model.state_dict()

    pretrained_cla_weight_key = []
    pretrained_cla_weight_value = []
    model_weight_key = []
    model_weight_value = []

    for k, v in pretrained_cla_dict.items():
        # print(k)
        pretrained_cla_weight_key.append(k)
        pretrained_cla_weight_value.append(v)

    pretrained_cla_weight_key = pretrained_cla_weight_key[1:-2]
    pretrained_cla_weight_value = pretrained_cla_weight_value[1:-2]

    for k, v in model_dict.items():
        # print(k)
        model_weight_key.append(k)
        model_weight_value.append(v)
    # print(model_weight_key)
    new_dict = {}
    if round == 1:
        for i in range(6 + 4 * 6 * block + 6 * (block - 1)):
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i]
    else:
        pretrained_ae_dict = torch.load(best_ae_model_path)
        pretrained_ae_weight_key = []
        pretrained_ae_weight_value = []

        for k, v in pretrained_ae_dict.items():
            # print(k)
            pretrained_ae_weight_key.append(k)
            pretrained_ae_weight_value.append(v)

        for i in range(len(model_weight_key)):
            if i < 6 + 4 * 6 * block + 6 * (block - 1):
                new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i]
            else:
                new_dict[model_weight_key[i]] = pretrained_ae_weight_value[i]
            # new_dict[model_weight_key[i]] = pretrained_ae_weight_value[i]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # criterion
    criterion = nn.MSELoss()

    batchShape_clean = [args.batch_size, 1, args.image_size, args.image_size]
    batchShape_adv = [args.batch_size, 1, args.image_size, args.image_size]

    # initialize figure
    f, a = plt.subplots(2, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    min_loss = 10
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
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

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
                          epoch + 1, args.epochs, batch_id, (60000 / args.batch_size) + 1, sum_loss / batch_id, args.lr,
                          (end - start)))
                    f1.write("[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs"
                             % (epoch + 1, args.epochs, batch_id, (60000 / args.batch_size) + 1, sum_loss / batch_id,
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
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                        sum_loss_test += loss.cpu().data.numpy()

                        if batch_id_test == 4:
                            # 观察原图以及去噪后的图片
                            # first row:clean images
                            for i in range(5):
                                a[0][i].clear()
                                a[0][i].imshow(np.reshape(inputs.cpu().data.numpy()[i, 0, :, :], (28, 28)),
                                               cmap="gray")
                                a[0][i].set_xticks(())
                                a[0][i].set_yticks(())
                                a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")

                            # second row:decoded images
                            for i in range(5):
                                a[1][i].clear()
                                a[1][i].imshow(np.reshape(outputs.cpu().data.numpy()[i], (28, 28)), cmap="gray")
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





