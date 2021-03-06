import numpy as np
import argparse
from torch import nn
import torch
import torchvision
from torch.autograd import Variable
from tqdm import *
from torchvision import transforms
from models import resnet
import matplotlib.pyplot as plt
from models.joint_training import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder


def robutness(source_cla_model_path, test_cla_model_path, eps, model_type, id):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument("--input_path", type=str, default="D:/python_workplace/resnet-AE/inputData/mnist/",
                        help="data set dir path")
    parser.add_argument("--epsilon", type=float, default=eps, help="Maximum size of adversarial perturbation.")
    parser.add_argument("--image_size", type=int, default=28, help="Width of each input images.")
    parser.add_argument("--batch_size", type=int, default=500, help="How many images process at one time.")
    parser.add_argument("--num_classes", type=int, default=10, help="num classes")
    parser.add_argument("--output_path_acc", type=str,
                        default="D:/python_workplace/resnet-AE/test/Model_Robustness/acc_" + model_type +
                                ".txt",
                        help="Output directory with acc file.")
    parser.add_argument("--output_path_image", type=str,
                        default="D:/python_workplace/resnet-AE/test/Model_Robustness/image/" + model_type + "/",
                        help="Output directory with image file.")

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

    # Load pre-trained weights
    pretrained_cla_dict = torch.load(source_cla_model_path)
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
    if len(pretrained_cla_weight_key) == len(model_weight_value):
        model.load_state_dict(torch.load(source_cla_model_path))
        print("Weights Loaded!")
        model.to(device)
    else:
        new_dict = {}
        for i in range(len(model_dict)):
            new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i + 1]
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print("Weights Loaded!")
        model.to(device)

    # # test model on adversarial examples
    # model_test = resnet.resnet18(pretrained=False)

    model_test = joint_model_4(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])

    pretrained_cla_dict = torch.load(test_cla_model_path)
    model_test_dict = model_test.state_dict()

    pretrained_cla_weight_key = []
    pretrained_cla_weight_value = []
    model_test_weight_key = []
    model_test_weight_value = []

    for k, v in pretrained_cla_dict.items():
        # print(k)
        pretrained_cla_weight_key.append(k)
        pretrained_cla_weight_value.append(v)

    for k, v in model_test_dict.items():
        # print(k)
        model_test_weight_key.append(k)
        model_test_weight_value.append(v)
    if len(pretrained_cla_weight_key) == len(model_test_weight_value):
        model_test.load_state_dict(torch.load(test_cla_model_path))
        print("Weights Loaded!")
        model_test.to(device)
    else:
        new_dict = {}
        for i in range(len(model_test_dict)):
            new_dict[model_test_weight_key[i]] = pretrained_cla_weight_value[i]
        model_test_dict.update(new_dict)
        model_test.load_state_dict(model_test_dict)
        print("Weights Loaded!")
        model_test.to(device)


    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

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

        # # Classification before Adv
        # _, y_pred = torch.max(outputs.data, 1)
        # y_correct_test += y_pred.eq(y_true.data).cpu().sum().item()

        # Generate Adversarial Image
        # Add perturbation
        epsilon = args.epsilon
        x_grad = torch.sign(x_input.grad.data)
        x_adversarial = torch.clamp(x_input.data + epsilon * x_grad, 0, 1).to(device)
        # x_adversarial = (x_input.data + epsilon * x_grad).to(device)

        image_origin_test = x_input.cpu().data.numpy() * 255
        image_origin_test = np.rint(image_origin_test).astype(np.int)

        image_adversarial_test = x_adversarial.cpu().data.numpy() * 255
        image_adversarial_test = np.rint(image_adversarial_test).astype(np.int)

        noise_test = image_adversarial_test - image_origin_test
        # noise_test = np.where(noise_test >= args.L_F, args.L_F, noise_test)
        # noise_test = np.where(noise_test <= -args.L_F, args.L_F, noise_test)

        image_adversarial_test = noise_test + image_origin_test

        image_adversarial_test = image_adversarial_test / 255
        noise_test = noise_test / 255

        # Classification before Adv
        model_test.eval()
        outputs_test = model_test(x_input)
        _, y_pred = torch.max(outputs_test.data, 1)
        y_correct_test += y_pred.eq(y_true.data).cpu().sum().item()

        # Classification after optimization
        outputs_adversarial = model_test(
            Variable(torch.from_numpy(image_adversarial_test).type(torch.FloatTensor).to(device)))
        _, y_pred_adversarial = torch.max(outputs_adversarial.data, 1)
        y_correct_test_adversarial += y_pred_adversarial.eq(y_true.data).cpu().sum().item()

        y_preds_test.extend(list(y_pred.cpu().data.numpy()))
        y_preds_test_adversarial.extend(list(y_pred_adversarial.cpu().data.numpy()))
        noises_test.extend(list(noise_test))
        images_adv_test.extend(list(image_adversarial_test))
        images_clean_test.extend(list(x_input.cpu().data.numpy()))
        y_trues_clean_test.extend(list(y_true.cpu().data.numpy()))

        # print(noises_test)

    idxs = np.random.choice(range(10000), size=(20,), replace=False)
    for matidx, idx in enumerate(idxs):
        orig_im = images_clean_test[idx].reshape(28, 28)
        adv_im = orig_im + noises_test[idx].reshape(28, 28)
        disp_im = np.concatenate((orig_im, adv_im), axis=1)
        plt.subplot(5, 4, matidx + 1)
        plt.imshow(disp_im, "gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Orig: {} | Adv: {}".format(y_preds_test[idx], y_preds_test_adversarial[idx]))
    plt.suptitle("Epsilon:" + str(eps))
    plt.draw()
    plt.pause(0.05)
    img = plt.gcf()
    image_path_name = args.output_path_image + str(id) + "_" + str(eps) + ".png"
    img.savefig(image_path_name)
    plt.show()

    # print(y_preds_test)
    total_images_test = len(test_datasets)
    acc_test_clean = y_correct_test / total_images_test * 100
    acc_test_adv = y_correct_test_adversarial / total_images_test * 100
    print("Model Type: %s | Epsilon: %.2f" % (model_type, eps))
    print("Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%" % (acc_test_clean, acc_test_adv))
    print("Adversarial Test Set Total Misclassified: %d" % (total_images_test - y_correct_test_adversarial))

    with open(args.output_path_acc, "a+") as f1:
        f1.write("Epsilon: %.2f | Test Set Accuracy Before: %.2f%% | Accuracy After: %.2f%%"
                 % (eps, acc_test_clean, acc_test_adv))
        f1.write("\n")


if __name__ == "__main__":
    source_model = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_69.pth"
    adv_FGSM_training = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/model/model_87.pth"
    adv_DDN_training = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/model/model_87.pth"
    joint_train = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/mnist/block_4/model/model_44.pth"
    paths = [source_model, adv_FGSM_training, adv_DDN_training, joint_train]
    model_name = ["Source Model", "Adv FGSM Model", "Adv DDN Model", "Joint Model"]
    plt.ion()
    for i in range(1):
        source_cla_model_path = paths[0]
        test_cla_model_path = paths[i + 3]
        model_type = model_name[i + 3]
        for j in range(20):
            eps = round(0.05 * (j + 1), 2)
            # print(eps)
            robutness(source_cla_model_path, test_cla_model_path, eps, model_type, j + 1)
