import argparse
import os
import time
import torch
from torch import nn, optim
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from torchsummary import summary
from models.joint_training_cifar10 import joint_model_1, joint_model_2, joint_model_3, \
    BasicBlock_encoder, BasicBlock_decoder, joint_model_block2_1, joint_model_block3_1, joint_model_block4_1
from models import resnet_cifar
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
    labels_list = data_dict["labels"]
    images_adv_list = data_dict["images_adv"]

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


def adversarial_learning(best_cla_model_path, adv_example_path):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    parser = argparse.ArgumentParser("Image classifical!")
    parser.add_argument("--epochs", type=int, default=200, help="Epoch default:50.")
    parser.add_argument("--image_size", type=int, default=32, help="Image Size default:28.")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch_size default:256.")
    parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    args = parser.parse_args()

    # Load model
    model = resnet_cifar.resnet18(pretrained=False)
    model.to(device)
    # summary(model,(3,32,32))
    # print(model)

    # Load pre-trained weights
    # image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    # image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
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

    for k, v in model_dict.items():
        # print(k)
        model_weight_key.append(k)
        model_weight_value.append(v)

    new_dict = {}
    for i in range(len(model_dict)):
        new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i + 2]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    # model = NormalizedModel(model=model, mean=image_mean, std=image_std)

    # model.load_state_dict(torch.load(best_cla_model_path))
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # batch_shape
    batchShape_adv = [args.batch_size, 3, args.image_size, args.image_size]
    batchShape_clean = [args.batch_size, 3, args.image_size, args.image_size]

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
            # print(labels[0:20])
            total += len(images_clean)
            images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
            images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            model.to(device)
            # 测试clean数据集的测试集
            outputs_clean = model(images_clean)
            _, predicted_clean = torch.max(outputs_clean.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_clean += (predicted_clean == labels).sum().item()
            # 测试adv数据集的测试集
            outputs_adv = model(images_adv)
            _, predicted_adv = torch.max(outputs_adv.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            correct_adv += (predicted_adv == labels).sum().item()
        # print(total)
        # print(correct_clean)
        # print(total_train)
        acc_clean = correct_clean / total * 100
        acc_adv = correct_adv / total * 100
        print("Clean Test Set Accuracy：%.2f%%" % acc_clean)
        print("Adv Test Set Accuracy：%.2f%%" % acc_adv)


# def adversarial_learning(best_cla_model_path, adv_example_path):
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(device)
#
#     parser = argparse.ArgumentParser("Image classifical!")
#     parser.add_argument("--epochs", type=int, default=200, help="Epoch default:50.")
#     parser.add_argument("--image_size", type=int, default=32, help="Image Size default:28.")
#     parser.add_argument("--batch_size", type=int, default=200, help="Batch_size default:256.")
#     parser.add_argument("--lr", type=float, default=0.01, help="learing_rate. Default=0.01")
#     parser.add_argument("--num_classes", type=int, default=10, help="num classes")
#     args = parser.parse_args()
#
#     # Load model
#     model = joint_model_2(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
#     model.to(device)
#     # summary(model,(3,32,32))
#     # print(model)
#
#     # Load pre-trained weights
#     model.load_state_dict(torch.load(best_cla_model_path))
#     model.to(device)
#
#     # criterion
#     criterion = nn.CrossEntropyLoss().to(device)
#
#     # batch_shape
#     batchShape_adv = [args.batch_size, 3, args.image_size, args.image_size]
#     batchShape_clean = [args.batch_size, 3, args.image_size, args.image_size]
#
#     print("Waiting for Testing!")
#     with torch.no_grad():
#         # 测试clean test set
#         total = 0
#         correct_clean = 0
#         correct_adv = 0
#         for batchSize, images_clean, images_adv, labels in load_images(adv_example_path,
#                                                                        batchShape_clean,
#                                                                        batchShape_adv):
#             model.eval()
#             # print(labels[0:20])
#             total += len(images_clean)
#             images_clean = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
#             images_adv = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
#             labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
#             # model.noise = nn.Parameter(data=torch.zeros([len(labels), 1, 28, 28]), requires_grad=False)
#             model.to(device)
#             # 测试clean数据集的测试集
#             outputs_clean_cla, outputs_clean_ae = model(images_clean)
#             _, predicted_clean_cla = torch.max(outputs_clean_cla.data, 1)
#             correct_clean += (predicted_clean_cla == labels).sum().item()
#             # 测试adv数据集的测试集
#             outputs_adv_cla, outputs_adv_ae = model(images_adv)
#             _, predicted_adv_cla = torch.max(outputs_adv_cla.data, 1)
#             correct_adv += (predicted_adv_cla == labels).sum().item()
#         # print(total)
#         # print(correct_clean)
#         # model.noise = nn.Parameter(data=torch.zeros([256, 1, 28, 28]), requires_grad=False)
#         # print(total_train)
#         acc_clean = correct_clean / total * 100
#         acc_adv = correct_adv / total * 100
#         print("Clean Test Set Accuracy：%.2f%%" % acc_clean)
#         print("Adv Test Set Accuracy：%.2f%%" % acc_adv)


if __name__ == "__main__":
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_282.pth"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/cifar10/model/model_50.pth"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/ResNet18/cifar10/block_2/round_1/model/model_21.pth"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/cifar10/block_2/round_3/model/model_41.pth"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_6/model/model_100.pth"
    # best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_3/model/model_50.pth"
    best_cla_model_path = "D:/python_workplace/resnet-AE/model/DDN_adv/cifar10_230.pth"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/CW/cifar10/test/test.pkl"
    # adv_example_path = "D:/python_workplace/resnet-AE/outputData/DDN/cifar10/test/test.pkl"
    adv_example_path = "D:/python_workplace/resnet-AE/outputData/FGSM/cifar10/test/test.pkl"

    adversarial_learning(best_cla_model_path, adv_example_path)