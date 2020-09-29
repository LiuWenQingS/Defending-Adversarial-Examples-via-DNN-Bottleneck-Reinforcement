import argparse
import os
import time
import torch
from torch import nn, optim
from torchsummary import summary
from models import resnet
import numpy as np
import pickle
from models import ComDefend


def load_images_1(input_dir, batch_shape_clean, batch_shape_adv):
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
        image_clean[idx, 0, :, :] = images_clean_list[j]
        image_adv[idx, 0, :, :] = images_adv_list[j]
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


def load_images_2(input_dir, batch_shape_clean, batch_shape_adv):
    image_clean = np.zeros(batch_shape_clean)
    image_adv = np.zeros(batch_shape_adv)
    labels = np.zeros(batch_shape_clean[0])
    idx = 0
    batch_size = batch_shape_adv[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_clean_list = data_dict["images"]
    noises = data_dict["noises_test"]
    labels_list = data_dict["labels"]
    # y_pred_test = data_dict["y_pred_test"]
    # print(labels_list[0:20])
    # print(y_pred_test[0:20])
    images_adv_list = []

    for i in range(len(labels_list)):
        images_clean_list[i] = images_clean_list[i].reshape(28, 28)
        # print(orig_im)
        images_adv_list.append(images_clean_list[i] + noises[i].reshape(28, 28))

    for j in range(len(images_clean_list)):
        image_clean[idx, 0, :, :] = images_clean_list[j]
        image_adv[idx, 0, :, :] = images_adv_list[j]
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


def test_mnist_1(best_cla_model_path, best_com_model_path, adv_example_path):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    parser = argparse.ArgumentParser("Image classifical!")
    parser.add_argument("--epochs", type=int, default=200, help="Epoch default:50.")
    parser.add_argument("--image_size", type=int, default=28, help="Image Size default:28.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    args = parser.parse_args()

    # Load model
    cla_model = resnet.resnet18(pretrained=False)
    com_model = ComDefend.ComDefend()

    # Load pre-trained weights
    pretrained_cla_dict = torch.load(best_cla_model_path)
    model_dict = cla_model.state_dict()

    pretrained_cla_weight_key = []
    pretrained_cla_weight_value = []
    model_weight_key = []
    model_weight_value = []

    for k, v in pretrained_cla_dict.items():
        pretrained_cla_weight_key.append(k)
        pretrained_cla_weight_value.append(v)

    for k, v in model_dict.items():
        model_weight_key.append(k)
        model_weight_value.append(v)

    new_dict = {}
    for i in range(len(model_dict)):
        new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i + 1]

    model_dict.update(new_dict)
    cla_model.load_state_dict(model_dict)

    # cla_model.load_state_dict(torch.load(best_cla_model_path))
    com_model.load_state_dict(torch.load(best_com_model_path))

    cla_model.to(device)
    com_model.to(device)

    # batch_shape
    batchShape_adv = [args.batch_size, 1, args.image_size, args.image_size]
    batchShape_clean = [args.batch_size, 1, args.image_size, args.image_size]

    print("Waiting for Testing!")
    with torch.no_grad():
        # 测试clean test set
        total = 0
        correct_clean = 0
        correct_adv = 0
        for batchSize, images_clean, images_adv, labels in load_images_2(adv_example_path, batchShape_clean, batchShape_adv):
            cla_model.eval()
            com_model.eval()

            total += len(images_clean)
            images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
            images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)

            com_images_clean = com_model(images_clean)
            com_images_adv = com_model(images_adv)

            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            cla_model.to(device)
            com_model.to(device)

            # 测试clean数据集的测试集
            outputs_clean = cla_model(com_images_clean)
            _, predicted_clean = torch.max(outputs_clean.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_clean += (predicted_clean == labels).sum().item()\

            # 测试adv数据集的测试集
            outputs_adv = cla_model(com_images_adv)
            _, predicted_adv = torch.max(outputs_adv.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_adv += (predicted_adv == labels).sum().item()

        acc_clean = correct_clean / total * 100
        acc_adv = correct_adv / total * 100
        print("Clean Test Set Accuracy：%.2f%%" % acc_clean)
        print("Adv Test Set Accuracy：%.2f%%" % acc_adv)


if __name__ == "__main__":
    best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_69.pth"
    best_com_model_path = "D:/python_workplace/resnet-AE/checkpoint/Autoencoder/mnist/comdefend/model/model_300.pth"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/CW/mnist/test/test.pkl"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/DDN/mnist/test/test.pkl"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/FGSM/mnist/test/test.pkl"
    adv_example_path = "D:/python_workplace/resnet-AE/outputData/LBFGS/mnist/test/test.pkl"

    test_mnist_1(best_cla_model_path, best_com_model_path, adv_example_path)