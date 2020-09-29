import numpy as np
import argparse
from torch import nn
import torch
import torchvision
import random
from torch.autograd import Variable
from tqdm import *
from torchvision import transforms
from models import vgg
import pickle

torch.cuda.manual_seed(10)
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


def FGSM(best_cla_model_path, device_used):
    device = torch.device(device_used if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument("--input_path", type=str,
                        default="C:/Users/WenqingLiu/cifar/cifar10/cifar-10-batches-py/",
                        help="data set dir path")
    parser.add_argument("--output_path_train", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/data/train/train.pkl",
                        help="Output directory with train images.")
    parser.add_argument("--output_path_test", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/data/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon")
    parser.add_argument("--L_F", type=int, default=5, help="L_F")
    parser.add_argument("--image_size", type=int, default=32, help="Width of each input images.")
    parser.add_argument("--batch_size", type=int, default=200, help="How many images process at one time.")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--output_path_acc", type=str,
                        default="D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/data/acc.txt",
                        help="Output directory with acc file.")

    args = parser.parse_args()

    # Transform Init
    transform_train = transforms.Compose([
        # transforms.Resize(32),  # 将图像转化为32 * 32
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(32),  # 将图像转化为32 * 32
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data Parse
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

    # Define Network
    model = vgg.vgg16_bn(pretrained=False)

    # Load pre-trained weights
    # model.load_state_dict(torch.load(best_cla_model_path))
    # model.to(device)

    model_dict = model.state_dict()
    pretrained_ae_model = torch.load(best_cla_model_path)
    model_key = []
    model_value = []
    pretrained_ae_key = []
    pretrained_ae_value = []
    for k, v in model_dict.items():
        # print(k)
        model_key.append(k)
        model_value.append(v)
    for k, v in pretrained_ae_model.items():
        # print(k)
        pretrained_ae_key.append(k)
        pretrained_ae_value.append(v)
    new_dict = {}

    for i in range(len(model_dict)):
        new_dict[model_key[i]] = pretrained_ae_value[i]

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    print("Weights Loaded!")

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
        epsilon = args.epsilon
        x_grad = torch.sign(x_input.grad.data)
        x_adversarial = torch.clamp(x_input.data + epsilon * x_grad, 0, 1).to(device)
        # x_adversarial = (x_input.data + epsilon * x_grad).to(device)
        image_adversarial_train = x_adversarial
        noise_train = x_adversarial - x_input

        # image_origin_train = x_input.cpu().data.numpy() * 255
        # # print(x_input.cpu().data.numpy().shape)
        # image_origin_train = np.rint(image_origin_train).astype(np.int)
        # 
        # image_adversarial_train = x_adversarial.cpu().data.numpy() * 255
        # image_adversarial_train = np.rint(image_adversarial_train).astype(np.int)
        # 
        # noise_train = image_adversarial_train - image_origin_train
        # # noise_train = np.where(noise_train >= args.L_F, args.L_F, noise_train)
        # # noise_train = np.where(noise_train <= -args.L_F, args.L_F, noise_train)
        # 
        # image_adversarial_train = noise_train + image_origin_train
        # 
        # noise_train = noise_train / 255
        # image_adversarial_train = image_adversarial_train / 255

        # Classification after optimization
        # outputs_adversarial = model(Variable(torch.from_numpy(image_adversarial_train).type(torch.FloatTensor).to(device)))
        outputs_adversarial = model(image_adversarial_train)
        _, y_pred_adversarial = torch.max(outputs_adversarial.data, 1)
        y_correct_train_adversarial += y_pred_adversarial.eq(y_true.data).cpu().sum().item()

        y_preds_train.extend(list(y_pred.cpu().data.numpy()))
        y_preds_train_adversarial.extend(list(y_pred_adversarial.cpu().data.numpy()))
        # noises_train.extend(list(noise_train))
        # images_adv_train.extend(list(image_adversarial_train)
        noises_train.extend(list(noise_train.cpu().data.numpy()))
        images_adv_train.extend(list(image_adversarial_train.cpu().data.numpy()))
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
        epsilon = args.epsilon
        x_grad = torch.sign(x_input.grad.data)
        x_adversarial = torch.clamp(x_input.data + epsilon * x_grad, 0, 1).to(device)
        # x_adversarial = (x_input.data + epsilon * x_grad).to(device)
        image_adversarial_test = x_adversarial
        noise_test = x_adversarial - x_input

        # image_origin_test = x_input.cpu().data.numpy() * 255
        # image_origin_test = np.rint(image_origin_test).astype(np.int)
        #
        # image_adversarial_test = x_adversarial.cpu().data.numpy() * 255
        # image_adversarial_test = np.rint(image_adversarial_test).astype(np.int)
        #
        # noise_test = image_adversarial_test - image_origin_test
        # # noise_test = np.where(noise_test >= args.L_F, args.L_F, noise_test)
        # # noise_test = np.where(noise_test <= -args.L_F, args.L_F, noise_test)
        #
        # image_adversarial_test = noise_test + image_origin_test
        #
        # image_adversarial_test = image_adversarial_test / 255
        # noise_test = noise_test / 255

        # Classification after optimization
        # outputs_adversarial = model(Variable(torch.from_numpy(image_adversarial_test).type(torch.FloatTensor).to(device)))
        outputs_adversarial = model(image_adversarial_test)
        _, y_pred_adversarial = torch.max(outputs_adversarial.data, 1)
        y_correct_test_adversarial += y_pred_adversarial.eq(y_true.data).cpu().sum().item()

        y_preds_test.extend(list(y_pred.cpu().data.numpy()))
        y_preds_test_adversarial.extend(list(y_pred_adversarial.cpu().data.numpy()))
        # noises_test.extend(list(noise_test))
        # images_adv_test.extend(list(image_adversarial_test))
        noises_test.extend(list(noise_test.cpu().data.numpy()))
        images_adv_test.extend(list(image_adversarial_test.cpu().data.numpy()))
        images_clean_test.extend(list(x_input.cpu().data.numpy()))
        y_trues_clean_test.extend(list(y_true.cpu().data.numpy()))

        # print(noises_test)

    total_images_test = len(test_datasets)
    acc_test_clean = y_correct_test / total_images_test * 100
    acc_test_adv = y_correct_test_adversarial / total_images_test * 100
    total_images_train = len(train_datasets)
    acc_train_clean = y_correct_train / total_images_train * 100
    acc_train_adv = y_correct_train_adversarial / total_images_train * 100

    print("Train Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%" % (acc_train_clean, acc_train_adv))
    print("Train Set Total misclassification: %d" % (total_images_train - y_correct_train_adversarial))

    print("Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%" % (acc_test_clean, acc_test_adv))
    print("Test Set Total misclassification: %d" % (total_images_test - y_correct_test_adversarial))

    with open(args.output_path_acc, "w") as f1:
        f1.write("Train Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%"
                 % (acc_train_clean, acc_train_adv))
        f1.write("\n")
        f1.write("Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%"
                 % (acc_test_clean, acc_test_adv))

    with open(args.output_path_train, "wb") as f2:
        adv_data_dict = {
            "images_clean": images_clean_train,
            "images_adv": images_adv_train,
            "labels": y_trues_clean_train,
            "y_preds": y_preds_train,
            "noises": noises_train,
            "y_preds_adversarial": y_preds_train_adversarial,
        }
        pickle.dump(adv_data_dict, f2)

    with open(args.output_path_test, "wb") as f3:
        adv_data_dict = {
            "images_clean": images_clean_test,
            "images_adv": images_adv_test,
            "labels": y_trues_clean_test,
            "y_preds": y_preds_test,
            "noises": noises_test,
            "y_preds_adversarial": y_preds_test_adversarial,
        }
        pickle.dump(adv_data_dict, f3)


if __name__ == "__main__":
    FGSM("D:/python_workplace/resnet-AE/checkpoint/Classification/VGG/cifar10/model/model_282.pth",  "cuda:0")
    # FGSM("D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/model/model_10.pth",  "cuda:0")
    # FGSM("D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/model/model_100.pth", "cuda:0")
    # FGSM("D:/python_workplace/resnet-AE/checkpoint/Joint_Training/VGG/cifar10/model/model_200.pth", "cuda:0")
    # FGSM("D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/VGG/cifar10/model/model_150.pth", "cuda:0")