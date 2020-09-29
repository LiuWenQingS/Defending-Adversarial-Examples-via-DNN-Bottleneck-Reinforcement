import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle


class simpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AE_simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2):
        super(AE_simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)

        self.layer3 = nn.Linear(n_hidden_2, n_hidden_1)
        self.layer4 = nn.Linear(n_hidden_1, in_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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
        image_clean[idx, :] = images_clean_list[j]
        image_adv[idx, :] = images_adv_list[j]
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


def cla(model, FCSize):
    # 定义一些超参数
    batch_size = 512
    learning_rate = 0.001
    epoch = 50
    # path
    acc_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/Size_" \
                    + str(FCSize) + "/acc.txt"
    best_acc_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/Size_" \
                         + str(FCSize) + "/best_acc.txt"
    log_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/Size_" \
                    + str(FCSize) + "/log.txt"
    model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/Size_" \
                 + str(FCSize) + "/model/"

    # transform
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=True, transform=data_tf,
                                   download=True)
    test_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    best_acc = 70
    best_epoch = 0
    # 训练模型
    print("Start Training FCNet!")
    with open(acc_file_path, "w") as f1:
        with open(log_file_path, "w")as f2:
            for i in range(epoch):
                # if epoch + 1 <= 50:
                #     args.lr = 0.1
                # elif epoch + 1 > 50 & epoch + 1 <= 100:
                #     args.lr = 0.01
                # else:
                #     args.lr = 0.001
                #
                # # Optimization
                # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                sum_loss = 0
                correct_train = 0
                total_train = 0
                print('Epoch: %d' % (i + 1))
                for j, data in enumerate(train_loader, 0):
                    start = time.time()
                    # prepare data
                    img, labels = data
                    img = img.view(img.size(0), -1)
                    img = img.to(device)
                    labels = labels.to(device)

                    model.train()
                    optimizer.zero_grad()

                    # training model
                    outputs = model(img)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += predicted.eq(labels.data).cpu().sum().item()

                    end = time.time()

                    print('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs'
                          % (i + 1, epoch, j + 1, len(train_loader), sum_loss / (j + 1), correct_train / total_train * 100,
                             learning_rate,
                             (end - start)))
                    f2.write('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Acc: %.2f%% | Lr: %.04f | Time: %.03fs'
                             % (i + 1, epoch, j + 1, len(train_loader), sum_loss / (j + 1), correct_train / total_train * 100,
                                learning_rate,
                                (end - start)))
                    f2.write('\n')
                    f2.flush()

                # 模型评估
                model.eval()
                correct_test = 0
                total_test = 0
                for data in test_loader:
                    img, labels = data
                    img = img.view(img.size(0), -1)
                    img = img.to(device)
                    labels = labels.to(device)

                    outputs = model(img)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                print('Test Set Accuracy：%.2f%%' % (correct_test / total_test * 100))
                acc = correct_test / total_test * 100
                # 保存测试集准确率至acc.txt文件中
                f1.write("Epoch=%03d,Accuracy= %.2f%%" % (i + 1, acc))
                f1.write('\n')
                f1.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                if acc > best_acc:
                    best_acc = acc
                    f3 = open(best_acc_file_path, "w")
                    f3.write("Epoch=%d,best_acc= %.2f%%" % (i + 1, acc))
                    f3.close()
                    best_epoch = i + 1
                if i + 1 == epoch:
                    print('Saving model!')
                    torch.save(model.state_dict(), '%s/model_%d.pth' % (model_path, i + 1))
                    print('Model saved!')
            print("Training Finished, TotalEpoch = %d, BestEpoch = %d, Best Accuracy = %.2f%%" % (epoch, best_epoch,
                                                                                                  best_acc))
    return best_epoch


def FGSM(best_cla_model_path, model, FCSize, round, ae_training_set):
    # hyper-parameters
    batch_size = 1000
    epsilon = 0.2

    # file path
    output_path_train = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                        "/Size_" + str(FCSize) + "/round_" + str(round) + "/train/train.pkl"
    output_path_test = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                       "/Size_" + str(FCSize) + "/round_" + str(round) + "/test/test.pkl"
    output_path_acc = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                      "/Size_" + str(FCSize) + "/round_" + str(round) + "/acc.txt"

    # transform
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=True, transform=data_tf,
                                   download=True)
    test_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained weights
    model.load_state_dict(torch.load(best_cla_model_path))
    print("Weights Loaded!")
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # adversarial examples of train set
    noises_train = []
    y_preds_train = []
    y_preds_train_adversarial = []
    y_correct_train = 0
    y_correct_train_adversarial = 0
    images_clean_train = []
    images_adv_train = []
    y_trues_clean_train = []

    for data in train_loader:
        x_input, y_true = data
        x_input = x_input.view(x_input.size(0), -1)
        x_input, y_true = x_input.to(device), y_true.to(device)
        x_input.requires_grad_()

        # Forward pass
        model.eval()
        outputs = model(x_input)
        loss = criterion(outputs, y_true)
        # print(y_true.cpu().data.numpy())
        loss.backward()  # obtain gradients on x

        # Classification before Adv
        _, y_pred = torch.max(outputs.data, 1)
        y_correct_train += y_pred.eq(y_true.data).cpu().sum().item()

        # Generate Adversarial Image
        # Add perturbation
        epsilon = epsilon
        x_grad = torch.sign(x_input.grad.data)
        # x_adversarial = torch.clamp(x_input.data + epsilon * x_grad, 0, 1).to(device)
        x_adversarial = (x_input.data + epsilon * x_grad).to(device)

        # Classification after optimization
        outputs_adversarial = model(Variable(x_adversarial))
        _, y_pred_adversarial = torch.max(outputs_adversarial.data, 1)
        y_correct_train_adversarial += y_pred_adversarial.eq(y_true.data).cpu().sum().item()

        y_preds_train.extend(list(y_pred.cpu().data.numpy()))
        y_preds_train_adversarial.extend(list(y_pred_adversarial.cpu().data.numpy()))
        noises_train.extend(list((x_adversarial - x_input.data).cpu().numpy()))
        images_adv_train.extend(list(x_adversarial.cpu().numpy()))
        images_clean_train.extend(list(x_input.cpu().data.numpy()))
        y_trues_clean_train.extend(list(y_true.cpu().data.numpy()))

        # print(x_input.data.cpu().numpy())
        # print(noises_train)

    # adversarial examples of test set
    noises_test = []
    y_preds_test = []
    y_preds_test_adversarial = []
    y_correct_test = 0
    y_correct_test_adversarial = 0
    images_adv_test = []
    images_clean_test = []
    y_trues_clean_test = []

    for data in test_loader:
        x_input, y_true = data
        x_input = x_input.view(x_input.size(0), -1)
        x_input, y_true = x_input.to(device), y_true.to(device)
        x_input.requires_grad_()

        # Forward pass
        model.eval()
        outputs = model(x_input)
        loss = criterion(outputs, y_true)
        # print(y_true.cpu().data.numpy())
        loss.backward()  # obtain gradients on x

        # Classification before Adv
        _, y_pred = torch.max(outputs.data, 1)
        y_correct_test += y_pred.eq(y_true.data).cpu().sum().item()

        # Generate Adversarial Image
        # Add perturbation
        epsilon = epsilon
        x_grad = torch.sign(x_input.grad.data)
        # x_adversarial = torch.clamp(x_input.data + epsilon * x_grad, 0, 1).to(device)
        x_adversarial = (x_input.data + epsilon * x_grad).to(device)

        # Classification after optimization
        outputs_adversarial = model(Variable(x_adversarial))
        _, y_pred_adversarial = torch.max(outputs_adversarial.data, 1)
        y_correct_test_adversarial += y_pred_adversarial.eq(y_true.data).cpu().sum().item()

        y_preds_test.extend(list(y_pred.cpu().data.numpy()))
        y_preds_test_adversarial.extend(list(y_pred_adversarial.cpu().data.numpy()))
        noises_test.extend(list((x_adversarial - x_input.data).cpu().numpy()))
        images_adv_test.extend(list(x_adversarial.cpu().numpy()))
        images_clean_test.extend(list(x_input.cpu().data.numpy()))
        y_trues_clean_test.extend(list(y_true.cpu().data.numpy()))

        # print(noises_test)

    # idxs = np.random.choice(range(10000), size=(20,), replace=False)
    # for matidx, idx in enumerate(idxs):
    #     orig_im = images_clean_test[idx].reshape(28, 28)
    #     adv_im = orig_im + noises_test[idx].reshape(28, 28)
    #     disp_im = np.concatenate((orig_im, adv_im), axis=1)
    #     plt.subplot(5, 4, matidx + 1)
    #     plt.imshow(disp_im, "gray")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title("Orig: {} | Adv: {}".format(y_preds_test[idx], y_preds_test_adversarial[idx]))
    # plt.suptitle("Epsilon:" + str(epsilon))
    # plt.show()

    total_images_test = len(test_dataset)
    acc_test_clean = y_correct_test / total_images_test * 100
    acc_test_adv = y_correct_test_adversarial / total_images_test * 100
    total_images_train = len(train_dataset)
    acc_train_clean = y_correct_train / total_images_train * 100
    acc_train_adv = y_correct_train_adversarial / total_images_train * 100

    print("Train Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%" % (acc_train_clean, acc_train_adv))
    print("Train Set Total misclassification: %d" % (total_images_train - y_correct_train_adversarial))

    print("Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%" % (acc_test_clean, acc_test_adv))
    print("Test Set Total misclassification: %d" % (total_images_test - y_correct_test_adversarial))

    with open(output_path_acc, "w") as f1:
        f1.write("Train Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%"
                 % (acc_train_clean, acc_train_adv))
        f1.write('\n')
        f1.write("Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%"
                 % (acc_test_clean, acc_test_adv))

    with open(output_path_train, "wb") as f2:
        adv_data_dict = {
            "images_clean": images_clean_train,
            "images_adv": images_adv_train,
            "labels": y_trues_clean_train,
            "y_preds": y_preds_train,
            "noises": noises_train,
            "y_preds_adversarial": y_preds_train_adversarial,
        }
        pickle.dump(adv_data_dict, f2)

    with open(output_path_test, "wb") as f3:
        adv_data_dict = {
            "images_clean": images_clean_test,
            "images_adv": images_adv_test,
            "labels": y_trues_clean_test,
            "y_preds": y_preds_test,
            "noises": noises_test,
            "y_preds_adversarial": y_preds_test_adversarial,
        }
        pickle.dump(adv_data_dict, f3)


def ae_clean(best_cla_model_path, model, FCSize, round, ae_training_set):
    # hyper-parameters
    batch_size = 512
    epochs = 100

    # file path
    model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                 "/Size_" + str(FCSize) + "/round_" + str(round) + "/model/"
    image_compare = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/images/"
    log_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/log.txt"
    loss_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                     "/Size_" + str(FCSize) + "/round_" + str(round) + "/loss.txt"
    min_loss_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/min_loss.txt"

    # transform
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=True, transform=data_tf,
                                   download=True)
    test_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(len(model_weight_key))
    new_dict = {}
    for i in range(len(model_weight_key)):
        if i <= 3:
            new_dict[model_weight_key[i]] = pretrained_weight_value[i]
        # elif 3 < i <= 5:
        #     new_dict[model_weight_key[i]] = pretrained_weight_value[i - 2]
        # else:
        #     new_dict[model_weight_key[i]] = pretrained_weight_value[i - 6]

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

    min_loss = 10
    best_epoch = 1
    print("Start Training AutoEncoder!")
    with open(log_file_path, "w") as f1:
        with open(loss_file_path, "w") as f2:
            for epoch in range(0, epochs):
                if epoch + 1 <= 40:
                    lr = 0.01
                elif epoch + 1 > 40 & epoch + 1 <= 80:
                    lr = 0.001
                else:
                    lr = 0.0001

                # Optimization
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

                print('Epoch: %d' % (epoch + 1))
                sum_loss = 0.0
                batch_id = 1
                for i, data in enumerate(train_loader, 0):
                    start = time.time()

                    # model prepare
                    model.train()
                    optimizer.zero_grad()

                    # data prepare
                    inputs, labels = data
                    inputs = inputs.view(inputs.size(0), -1)
                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward + backward
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.cpu().data.numpy()

                    end = time.time()

                    print('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.04f | Time: %.03fs'
                          % (epoch + 1, epochs, batch_id, (60000 / batch_size) + 1, sum_loss / batch_id, lr,
                             (end - start)))
                    f1.write('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs'
                             % (epoch + 1, epochs, batch_id, (60000 / batch_size) + 1, sum_loss / batch_id, lr,
                                (end - start)))
                    f1.write('\n')
                    f1.flush()
                    batch_id += 1

                # 每训练完一个epoch测试loss
                print("Waiting for Testing!")
                with torch.no_grad():
                    sum_loss_test = 0
                    batch_id_test = 1
                    for i, data in enumerate(test_loader, 0):
                        model.eval()
                        inputs_s, labels = data
                        inputs = inputs_s.view(inputs_s.size(0), -1)
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                        sum_loss_test += loss.cpu().data.numpy()

                        if batch_id_test == 4:
                            # 观察原图以及去噪后的图片
                            # first row:clean images
                            for i in range(5):
                                a[0][i].clear()
                                a[0][i].imshow(np.reshape(inputs_s.cpu().data.numpy()[i, 0, :, :], (28, 28)),
                                               cmap='gray')
                                a[0][i].set_xticks(())
                                a[0][i].set_yticks(())
                                a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")

                            # second row:decoded images
                            for i in range(5):
                                a[1][i].clear()
                                a[1][i].imshow(np.reshape(outputs.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                                a[1][i].set_xticks(())
                                a[1][i].set_yticks(())
                                a[1][i].set_title("Decoded Images" + "[" + str(i + 1) + "]")
                            plt.suptitle("Epoch:" + str(epoch + 1))
                            plt.draw()
                            plt.pause(0.05)
                            if (epoch + 1) % 1 == 0:
                                img = plt.gcf()
                                img.savefig(image_compare + str(epoch + 1) + '.png')
                            plt.show()

                        batch_id_test += 1
                    print('Test Set Loss：%.4f' % sum_loss_test)

                    # 保存测试集loss至loss.txt文件中
                    f2.write("Epoch=%03d,Loss= %.4f" % (epoch + 1, sum_loss_test))
                    f2.write('\n')
                    f2.flush()
                    # 记录最小测试集loss并写入min_loss.txt文件中
                    if sum_loss_test < min_loss:
                        min_loss = sum_loss_test
                        f3 = open(min_loss_path, "w")
                        f3.write("Epoch=%d,min_loss= %.4f" % (epoch + 1, sum_loss_test))
                        f3.close()
                        best_epoch = epoch + 1
                    if epoch + 1 == epochs:
                        print('Saving model!')
                        torch.save(model.state_dict(), '%s/model_%d.pth' % (model_path, epoch + 1))
                        print('Model saved!')
    return best_epoch


def ae_adv(best_cla_model_path, model, FCSize, round, ae_training_set):
    # hyper-parameters
    batch_size = 512
    epochs = 100

    # file path
    model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                 "/Size_" + str(FCSize) + "/round_" + str(round) + "/model/"
    image_compare = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/images/"
    log_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/log.txt"
    loss_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                     "/Size_" + str(FCSize) + "/round_" + str(round) + "/loss.txt"
    min_loss_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/AutoEncoder" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/min_loss.txt"
    input_dir_trainSet = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                         "/Size_" + str(FCSize) + "/round_" + str(round) + "/train/train.pkl"
    input_dir_testSet = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                        "/Size_" + str(FCSize) + "/round_" + str(round) + "/test/test.pkl"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for i in range(len(model_weight_key)):
        if i <= 3:
            new_dict[model_weight_key[i]] = pretrained_weight_value[i]

    # for k, v in new_dict.items():
    #     print(k)

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # criterion
    criterion = nn.MSELoss()

    batchShape_clean = [batch_size, 784]
    batchShape_adv = [batch_size, 784]

    # initialize figure
    f, a = plt.subplots(3, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    min_loss = 10
    best_epoch = 1
    print("Start Training AutoEncoder!")
    with open(log_file_path, "w") as f1:
        with open(loss_file_path, "w") as f2:
            for epoch in range(0, epochs):
                if epoch + 1 <= 40:
                    lr = 0.01
                elif epoch + 1 > 40 & epoch + 1 <= 80:
                    lr = 0.001
                else:
                    lr = 0.0001

                # Optimization
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

                print('Epoch: %d' % (epoch + 1))
                sum_loss = 0.0
                batch_id = 1
                for batchSize, images_clean, images_adv, labels in load_images(input_dir_trainSet, 
                                                                               batchShape_clean, batchShape_adv):
                    start = time.time()

                    # model prepare
                    model.train()
                    optimizer.zero_grad()

                    # data prepare
                    images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                    images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                    # images_clean = images_clean.view(images_clean.size(0), -1)
                    # images_adv = images_adv.view(images_adv.size(0), -1)

                    # forward + backward
                    outputs = model(images_adv)
                    loss = criterion(outputs, images_clean)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.cpu().data.numpy()

                    end = time.time()

                    print('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.04f | Time: %.03fs'
                          % (
                              epoch + 1, epochs, batch_id, (60000 / batch_size) + 1, sum_loss / batch_id,
                              lr,
                              (end - start)))
                    f1.write('[Epoch:%d/%d] | [Batch:%d/%d] | Loss: %.03f | Lr: %.4f | Time: %.3fs'
                             % (epoch + 1, epochs, batch_id, (60000 / batch_size) + 1, sum_loss / batch_id,
                                lr, (end - start)))
                    f1.write('\n')
                    f1.flush()
                    batch_id += 1

                # 每训练完一个epoch测试loss
                print("Waiting for Testing!")
                with torch.no_grad():
                    sum_loss_test = 0
                    batch_id_test = 1
                    for batchSize, images_clean, images_adv, labels in load_images(input_dir_testSet, 
                                                                                   batchShape_clean, batchShape_adv):
                        model.eval()
                        images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                        images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                        # images_clean = images_clean_s.view(images_clean_s.size(0), -1)
                        # images_adv = images_adv_s.view(images_adv_s.size(0), -1)
                        outputs = model(images_adv)
                        loss = criterion(outputs, images_clean)
                        sum_loss_test += loss.cpu().data.numpy()

                        if batch_id_test == 4:
                            # 观察原图以及去噪后的图片
                            # first row:clean images
                            for i in range(5):
                                a[0][i].clear()
                                a[0][i].imshow(np.reshape(images_clean.cpu().data.numpy()[i, :], (28, 28)),
                                               cmap='gray')
                                a[0][i].set_xticks(())
                                a[0][i].set_yticks(())
                                a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")
                            # second row:adversarial images
                            for i in range(5):
                                a[1][i].clear()
                                a[1][i].imshow(np.reshape(images_adv.cpu().data.numpy()[i, :], (28, 28)),
                                               cmap='gray')
                                a[1][i].set_xticks(())
                                a[1][i].set_yticks(())
                                a[1][i].set_title("Adversarial Images" + "[" + str(i + 1) + "]")
                            # third row:decoded images
                            for i in range(5):
                                a[2][i].clear()
                                a[2][i].imshow(np.reshape(outputs.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                                a[2][i].set_xticks(())
                                a[2][i].set_yticks(())
                                a[2][i].set_title("Decoded Images" + "[" + str(i + 1) + "]")
                            plt.suptitle("Epoch:" + str(epoch + 1))
                            plt.draw()
                            plt.pause(0.05)
                            if (epoch + 1) % 1 == 0:
                                img = plt.gcf()
                                img.savefig(image_compare + str(epoch + 1) + '.png')
                            plt.show()

                        batch_id_test += 1
                    print('Test Set Loss：%.4f' % sum_loss_test)

                    # 保存测试集loss至loss.txt文件中
                    f2.write("Epoch=%03d,Loss= %.4f" % (epoch + 1, sum_loss_test))
                    f2.write('\n')
                    f2.flush()
                    # 记录最小测试集loss并写入min_loss.txt文件中
                    if sum_loss_test < min_loss:
                        min_loss = sum_loss_test
                        f3 = open(min_loss_path, "w")
                        f3.write("Epoch=%d,min_loss= %.4f" % (epoch + 1, sum_loss_test))
                        f3.close()
                        best_epoch = epoch + 1
                    if epoch + 1 == epochs:
                        print('Saving model!')
                        torch.save(model.state_dict(), '%s/model_%d.pth' % (model_path, epoch + 1))
                        print('Model saved!')
    return best_epoch


def reTrain(best_cla_model_path, best_ae_epoch, model, FCSize, round, ae_training_set):
    # hyper-parameters
    batch_size = 512
    epochs = 100
    image_size = 28

    # file path
    model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/ReTrain" \
                 "/Size_" + str(FCSize) + "/round_" + str(round) + "/model/"
    acc_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/ReTrain" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/acc.txt"
    best_acc_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/ReTrain" \
                         "/Size_" + str(FCSize) + "/round_" + str(round) + "/best_acc.txt"
    log_file_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set + "/ReTrain" \
                    "/Size_" + str(FCSize) + "/round_" + str(round) + "/log.txt"
    input_dir_trainSet = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                         "/Size_" + str(FCSize) + "/round_" + str(round) + "/train/train.pkl"
    input_dir_testSet = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/" + ae_training_set + \
                        "/Size_" + str(FCSize) + "/round_" + str(round) + "/test/test.pkl"

    # ae model path
    best_ae_model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/" + ae_training_set +  \
                         "/AutoEncoder/Size_" + str(FCSize) + "/round_" + str(round) + "/model/model_" \
                         + str(best_ae_epoch) + ".pth"

    # transform
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=True, transform=data_tf,
                                   download=True)
    test_dataset = datasets.MNIST(root='D:/python_workplace/resnet-AE/inputData/mnist/', train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_ae_dict = torch.load(best_ae_model_path)
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
        if i <= 3:
            new_dict[model_weight_key[i]] = pretrained_ae_weight_value[i]
        else:
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    batchShape_clean = [batch_size, 784]
    batchShape_adv = [batch_size, 784]

    print(f"Train numbers:{len(train_dataset)}")
    print(f"Test numbers:{len(test_dataset)}")

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # length_train = len(train_loader)
    best_acc_test_clean = 60
    best_acc_test_adv = 10
    best_acc_train_clean = 60
    best_acc_train_adv = 10
    best_epoch = 1
    flag_test = 0
    flag_train = 0
    print("Start Training FCNet After AutoEncoder!")
    with open(acc_file_path, "w") as f1:
        with open(log_file_path, "w")as f2:
            for epoch in range(0, epochs):
                if epoch + 1 <= 40:
                    lr = 0.01
                elif epoch + 1 > 40 & epoch + 1 <= 80:
                    lr = 0.001
                else:
                    lr = 0.0001

                # Optimization
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

                # 每个epoch之前测试一下准确率
                print("Waiting for Testing of Test Set!")
                with torch.no_grad():
                    correct_clean_test = 0
                    correct_adv_test = 0
                    total_test = 0
                    for batchSize, images_clean, images_adv, labels in load_images(input_dir_testSet,
                                                                                   batchShape_clean,
                                                                                   batchShape_adv):
                        model.eval()
                        images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                        images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                        # images_clean = images_clean.view(images_clean.size(0), -1)
                        # images_adv = images_adv.view(images_adv.size(0), -1)
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
                    print('Clean Test Set Accuracy：%.2f%%' % acc_clean_test)
                    print('Adv Test Set Accuracy：%.2f%%' % acc_adv_test)
                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Clean Test Set Accuracy= %.2f%%,Adv Test Set Accuracy = %.2f%%" % (
                        epoch, acc_clean_test, acc_adv_test))
                    f1.write('\n')
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
                    for batchSize, images_clean, images_adv, labels in load_images(input_dir_trainSet,
                                                                                   batchShape_clean,
                                                                                   batchShape_adv):
                        model.eval()
                        images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                        images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                        # images_clean = images_clean.view(images_clean.size(0), -1)
                        # images_adv = images_adv.view(images_adv.size(0), -1)
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
                    print('Clean Train Set Accuracy：%.2f%%' % acc_clean_train)
                    print('Adv Train Set Accuracy：%.2f%%' % acc_adv_train)
                    # 保存测试集准确率至acc.txt文件中
                    f1.write("Epoch=%03d,Clean Train Set Accuracy= %.2f%%,Adv Train Set Accuracy = %.2f%%" % (
                        epoch, acc_clean_train, acc_adv_train))
                    f1.write('\n')
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中并将准确率达标的模型保存
                    if acc_clean_train > best_acc_train_clean:
                        best_acc_train_clean = acc_clean_train
                    if acc_adv_train > best_acc_train_adv:
                        flag_train = 1
                        best_acc_train_adv = acc_adv_train

                if flag_train == 1 and flag_test == 1:
                    f3 = open(best_acc_file_path, "w")
                    f3.write("Epoch=%03d,Clean Test Set Accuracy= %.2f%%,Adv Test Set Accuracy = %.2f%%,"
                             "Clean Train Set Accuracy= %.2f%%,Adv Train Set Accuracy = %.2f%%"
                             % (epoch, acc_clean_test, acc_adv_test, acc_clean_train, acc_adv_train))
                    f3.close()

                    print('Saving model!')
                    torch.save(model.state_dict(), '%s/model_%d.pth' % (model_path, epoch))
                    print('Model saved!')
                    best_epoch = epoch

                flag_test = 0
                flag_train = 0

                print('Epoch: %d' % (epoch + 1))
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                start = time.time()
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    inputs, labels = data
                    inputs = inputs.view(inputs.size(0), -1)
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
                    # print(100.* correct / total)

                end = time.time()

                print('[Epoch:%d/%d] | Loss: %.03f | Test Acc of Clean Test Set: %.2f%% | '
                      'Test Acc of Clean Train Set: %.2f%% | Test Acc of Adv Test Set: %.2f%% | '
                      'Test Acc of Adv Train Set: %.2f%% | Train Acc: %.2f%% | Lr: %.04f | Time: %.03fs'
                      % (epoch + 1, epochs, sum_loss / (i + 1), acc_clean_test, acc_clean_train,
                         acc_adv_test, acc_adv_train, correct / total * 100, lr, (end - start)))
                f2.write('[Epoch:%d/%d] | Loss: %.03f | Test Acc of Clean Test Set: %.2f%% | '
                         'Test Acc of Clean Train Set: %.2f%% | Test Acc of Adv Test Set: %.2f%% | '
                         'Test Acc of Adv Train Set: %.2f%% | Train Acc: %.2f%% | Lr: %.04f | Time: %.03fs'
                         % (epoch + 1, epochs, sum_loss / (i + 1), acc_clean_test, acc_clean_train,
                            acc_adv_test, acc_adv_train, correct / total * 100, lr, (end - start)))
                f2.write('\n')
                f2.flush()
    return best_epoch


if __name__ == "__main__":
    # 5 rounds and 9 FC size
    for i in range(0, 1):
        # model
        model_cla = simpleNet(28 * 28, 1000, 100 * (i + 1), 10)
        model_ae = AE_simpleNet(28 * 28, 1000, 100 * (i + 1))
        # model = net.Activation_Net(28 * 28, 1000, 100 * (i + 1), 10)
        # model = net.Batch_Net(28 * 28, 1000, 100 * (i + 1), 10)
        # best_cla_epoch_s = cla(model_cla, 100 * (i + 1))
        best_cla_model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/Size_" \
                              + str(100 * (i + 1)) + "/model/model_" + str(50) + ".pth"
        # train autoencoder with clean set
        # for j in range(5):
        #     FGSM(best_cla_model_path, model_cla, 100 * (i + 1), j + 1, "AE_CleanSet")
        #     best_ae_epoch = ae_clean(best_cla_model_path, model_ae, 100 * (i + 1), j + 1, "AE_CleanSet")
        #     best_cla_epoch = reTrain(best_cla_model_path, 100, model_cla, 100 * (i + 1), j + 1, "AE_CleanSet")
        #     best_cla_model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/AE_CleanSet/ReTrain" \
        #                           "/Size_" + str(100 * (i + 1)) + "/round_" + str(j + 1) + "/model/model_" \
        #                           + str(best_cla_epoch) + ".pth"

        # train autoencoder with adv set
        for j in range(5):
            FGSM(best_cla_model_path, model_cla, 100 * (i + 1), j + 1, "AE_AdvSet")
            best_ae_epoch = ae_adv(best_cla_model_path, model_ae, 100 * (i + 1), j + 1, "AE_AdvSet")
            best_cla_epoch = reTrain(best_cla_model_path, 100, model_cla, 100 * (i + 1), j + 1, "AE_AdvSet")
            best_cla_model_path = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/AE_AdvSet/ReTrain" \
                                  "/Size_" + str(100 * (i + 1)) + "/round_" + str(j + 1) + "/model/model_" \
                                  + str(best_cla_epoch) + ".pth"