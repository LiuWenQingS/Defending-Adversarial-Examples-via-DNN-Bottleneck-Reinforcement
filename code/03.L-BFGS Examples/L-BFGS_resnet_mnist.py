import torch.optim as optim
import numpy as np
import argparse
from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import pickle
import time
from models import resnet
from models import ComDefend
from tqdm import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("Adversarial Examples")
parser.add_argument('--checkpoint_path', type=str,
                    default= 'D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_69.pth',
                    help='Path to checkpoint for nn network.')
parser.add_argument('--input_path', type=str, default='D:/python_workplace/resnet-AE/inputData/mnist/',
                    help='data set dir path')
parser.add_argument('--output_path_train', type=str,
                    default='D:/python_workplace/resnet-AE/outputData/LBFGS/mnist/train/train.pkl',
                    help='Output directory with train images.')
parser.add_argument('--output_path_test', type=str,
                    default='D:/python_workplace/resnet-AE/outputData/LBFGS/mnist/test/test.pkl',
                    help='Output directory with test images.')
parser.add_argument('--image_size', type=int, default=28, help='Width of each input images.')
parser.add_argument('--batch_size', type=int, default=1000, help='How many images process at one time.')
parser.add_argument('--num_classes', type=int, default=10, help='num classes')
parser.add_argument('--iter', type=int, default=1000, help='iters to optim')

args = parser.parse_args()

# Transform Init
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Data Parse
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


# Define Network
model = resnet.resnet18(pretrained=False)
com_model = ComDefend.ComDefend()

# Load pre-trained weights
best_com_model_path = "D:/python_workplace/resnet-AE/checkpoint/Autoencoder/mnist/comdefend/model/model_300.pth"
com_model.load_state_dict(torch.load(best_com_model_path))
com_model.to(device)
model.load_state_dict(torch.load(args.checkpoint_path))
model.to(device)
model.noise = nn.Parameter(data=torch.zeros([args.batch_size, 1, 28, 28]), requires_grad=True)
model.to(device)
# print(model)

# criterion
criterion = nn.CrossEntropyLoss().to(device)

# adversarial examples of test set
images_test, labels_test, images_clean_test, y_trues_test = [], [], [], []
noises_test, target_test, y_pred_test, y_pred_test_adversarial = [], [], [], []
y_correct_test, y_correct_test_adversarial, y_correct_test_adversarial_lastNoise = 0, 0, 0

for i, data in enumerate(test_loader, 0):
    model.eval()
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # print(labels.cpu().data.numpy())
    model.noise = nn.Parameter(data=torch.zeros([len(labels), 1, 28, 28]), requires_grad=True)
    model.to(device)

    outputs = model(images)

    # 原始分类类别
    _, y_pred = torch.max(outputs.data, 1)
    # print(y_pred.cpu().data.numpy())

    y_correct_test += y_pred.eq(labels.data).cpu().sum().item()

    # least_likely类别为target类别
    _, targets = torch.min(outputs.data, 1)

    y_pred_test.extend(list(y_pred.cpu().data.numpy()))
    target_test.extend(list(targets.cpu().data.numpy()))
    y_trues_test.extend(list(labels.cpu().data.numpy()))
    images_test.extend(list(images.cpu().data.numpy()))

model.noise = nn.Parameter(data=torch.zeros([1, 1, 28, 28]), requires_grad=True)

for i in range(len(test_datasets)):
    start = time.time()
    images = np.zeros([1, 1, args.image_size, args.image_size])
    images[0, :, :, :] = images_test[i]
    images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
    labels = Variable(torch.LongTensor(np.array([y_trues_test[i]])), requires_grad=False).to(device)
    target = Variable(torch.LongTensor(np.array([target_test[i]])), requires_grad=False).to(device)

    model.to(device)
    model.eval()

    com_model.eval()
    images = com_model(images)
    images = torch.from_numpy(images.cpu().data.numpy()).type(torch.FloatTensor).to(device)

    outputs = model(images)
    _, y = torch.max(outputs.data, 1)

    # print(labels)
    # print(target)
    # print(y)

    # Optimization Loop
    for iteration in range(args.iter):
        optimizer = optim.SGD(params=[model.noise], lr=0.1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target)

        # adv_loss = loss + torch.mean(torch.abs(model.noise))
        adv_loss = loss + torch.mean(torch.pow(model.noise, 2))
        # adv_loss = loss

        adv_loss.backward()
        optimizer.step()

        # noises = model.noise.data
        # noises = np.where(noises >= 0.1, 0.1, noises)
        # noises = np.where(noises <= -0.1, -0.1, noises)
        # model.noise = noises

        # print(adv_loss)
        # print(model.noise.cpu().data.numpy())

        # keep optimizing Until classif_op == target
        outputs_adversarial = model(images)
        _, y_adv = torch.max(outputs_adversarial.data, 1)
        # print(y_adv.cpu().data.numpy())
        # succeed = list((y_adv.cpu().data.numpy() == target.cpu().data.numpy())).count(False)
        # print(succeed)

        if y_adv.cpu().data.numpy() == target.cpu().data.numpy():
            break

        if iteration == args.iter - 1:
            print("Warning: optimization loop ran for %d iterations. The result may not be correct" % (args.iter))

    end = time.time()

    print("ImageId:[%d/%d] | Ground Truth: %d | Prediction Before: %d | Target: %d | Prediction After: %d | Time: %.3fs"
          %(i + 1, len(test_datasets), y_trues_test[i], y_pred_test[i], target_test[i], y_adv, (end - start)))

    images_clean_test.append(images.cpu().data.numpy())
    noises_test.append(model.noise.cpu().data.numpy())
    y_pred_test_adversarial.append(y_adv.cpu().data.numpy())
    y_correct_test_adversarial += y_adv.eq(labels.data).cpu().sum().item()
    # if (i + 1) % 20 == 0:
    #     break

# total_images_train = len(train_datasets)
# print("Train Set Accuracy Before: %.04f | Accuracy After: %.04f" %((y_correct_train / total_images_train),
#                                                                   (y_correct_train_adversarial / total_images_train)))

total_images_test = len(test_datasets)
print("Test Set Accuracy Before: %.04f | Accuracy After: %.04f" %((y_correct_test / total_images_test),
                                                                  (y_correct_test_adversarial / total_images_test)))
# with open(args.output_path_train, "wb") as f:
#     adv_data_dict = {
#         "images": images_clean_train,
#         "labels": y_trues_train,
#         "target": target_train,
#         "y_pred_train": y_pred_train,
#         "noises_train": noises_train,
#         "y_pred_train_adversarial": y_pred_train_adversarial,
#     }
#     pickle.dump(adv_data_dict, f)

# with open(args.output_path_test, "wb") as f:
#     adv_data_dict = {
#         "images": images_clean_test,
#         "labels": y_trues_test,
#         "target": target_test,
#         "y_pred_test": y_pred_test,
#         "noises_test": noises_test,
#         "y_pred_test_adversarial": y_pred_test_adversarial,
#     }
#     pickle.dump(adv_data_dict, f)