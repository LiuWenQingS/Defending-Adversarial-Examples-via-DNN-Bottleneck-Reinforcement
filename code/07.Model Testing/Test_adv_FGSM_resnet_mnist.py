import argparse
import os
import time
import torch
from torch import nn, optim
from torchsummary import summary
from models import resnet
import numpy as np
import pickle


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


def adversarial_learning(best_cla_model_path, adv_example_path):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    parser = argparse.ArgumentParser("Image classifical!")
    parser.add_argument("--epochs", type=int, default=200, help="Epoch default:50.")
    parser.add_argument("--image_size", type=int, default=28, help="Image Size default:28.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    args = parser.parse_args()

    # Load model
    model = resnet.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.avgpool = nn.AvgPool2d(1, 1)
    model.fc = nn.Linear(512*1, args.num_classes)
    model.to(device)
    # summary(model,(3,32,32))
    # print(model)

    # Load pre-trained weights
    model.load_state_dict(torch.load(best_cla_model_path))
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # batch_shape
    batchShape_adv = [args.batch_size, 1, args.image_size, args.image_size]
    batchShape_clean = [args.batch_size, 1, args.image_size, args.image_size]

    print("Waiting for Testing!")
    with torch.no_grad():
        # 测试clean test set
        total = 0
        correct_clean = 0
        correct_adv = 0
        for batchSize, images_clean, images_adv, labels in load_images(adv_example_path,
                                                                       batchShape_clean,
                                                                       batchShape_adv):
            model.eval()
            total += len(images_clean)
            images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
            images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            model.noise = nn.Parameter(data=torch.zeros([len(labels), 1, 28, 28]), requires_grad=False)
            model.to(device)
            # 测试clean数据集的测试集
            outputs_clean = model(images_clean)
            _, predicted_clean = torch.max(outputs_clean.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_clean += (predicted_clean == labels).sum().item()
            # 测试adv数据集的测试集
            outputs_adv = model(images_adv)
            _, predicted_adv = torch.max(outputs_adv.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_adv += (predicted_adv == labels).sum().item()
        print(total)
        print(correct_clean)
        model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
        # print(total_train)
        acc_clean = correct_clean / total * 100
        acc_adv = correct_adv / total * 100
        print("Clean Test Set Accuracy：%.2f%%" % acc_clean)
        print("Adv Test Set Accuracy：%.2f%%" % acc_adv)


if __name__ == "__main__":
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/mnist/block_1/round_10/model/model_30.pth"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/FGSM/mnist/AE_CleanSet/block_1/round_1/test/test.pkl"
    best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/ResNet18/mnist/block_4/round_4/model/model_4.pth"
    adv_example_path = "D:/python_workplace/resnet-AE/outputData/FGSM/mnist/AE_AdvSet/block_4/round_1/test/test.pkl"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/model/model_87.pth"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/FGSM/mnist/AE_CleanSet/block_1/round_1/test/test.pkl"

    adversarial_learning(best_cla_model_path, adv_example_path)