import argparse
import torch
import time
import foolbox
import pickle
from torch import nn
from models import resnet
from models import ComDefend
from torch.utils import data as data_load
from torchvision import datasets, transforms
from torchvision.utils import save_image

from fast_adversarial.fast_adv.models.mnist import SmallCNN
from fast_adversarial.fast_adv.attacks import DDN, CarliniWagnerL2
from fast_adversarial.fast_adv.utils import requires_grad_, l2_norm
# from models.joint_training import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder
# from models.joint_training_add import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder
# from models.joint_training_concat import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder
from models.joint_training_AddLoss import joint_model_1, joint_model_2, joint_model_3, joint_model_4, BasicBlock_encoder, BasicBlock_decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate adversarial examples on MNIST")
    parser.add_argument("--data_path", default="D:/python_workplace/resnet-AE/inputData/mnist/")
    parser.add_argument("--best_cla_model_path",
                        default="D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_69.pth")
    parser.add_argument("--best_com_model_path",
                        default="D:/python_workplace/resnet-AE/checkpoint/Autoencoder/mnist/comdefend/model/model_300.pth")
    parser.add_argument("--output_path_cw", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/CW/mnist/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_path_ddn_test", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/mnist/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_path_ddn_train", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/mnist/train/train.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_ddn", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/mnist/white_box_attack.txt",
                        help="Output directory with test images.")
    parser.add_argument("--output_cw", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/CW/mnist/white_box_attack.txt",
                        help="Output directory with test images.")

    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    source_model = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_69.pth"
    adv_training = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/mnist/model/model_87.pth"
    ae_clean_1 = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/ResNet18/mnist/block_4/round_6/model/model_2.pth"
    ae_clean_2 = "F:/AE_CleanSet/RetrainClassification/ResNet18/mnist/block_4/round_5/model/model_9.pth"
    ae_adv_1 = "E:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/mnist/block_4/round_7/model/model_4.pth"
    ae_adv_2 = "H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/mnist/block_3/round_6/model/model_1.pth"
    best_cla_model_path1 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/mnist/block_1/model/model_188.pth"
    best_cla_model_path2 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Add/ResNet18/mnist/block_1/model/model_3.pth"
    best_cla_model_path3 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Concat/ResNet18/mnist/block_1/model/model_3.pth"

    path_1 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_DCT/ResNet18/mnist/block_1/model/model_250.pth"
    path_2 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_DCT_0.01(14,7)/ResNet18/mnist/block_1/model/model_150.pth"
    path_3 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_DCT_0.2(14,7)/ResNet18/mnist/block_1/model/model_250.pth"
    path_4 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_DCT_0.1(14)/ResNet18/mnist/block_2/model/model_50.pth"
    path_5 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Subsample/ResNet18/mnist/block_4/model/model_50.pth"
    path_6 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Subsample_0.01(14,7)/ResNet18/mnist/block_1/model/model_50.pth"
    path_7 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Subsample_0.2(14,7)/ResNet18/mnist/block_1/model/model_50.pth"
    path_8 = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training_Subsample_0.1(14)/ResNet18/mnist/block_2/model/model_250.pth"

    paths = [source_model, adv_training, ae_clean_1, ae_clean_2, ae_adv_1, ae_adv_2, best_cla_model_path1,
             best_cla_model_path2, best_cla_model_path3]
    joint_paths = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8]

    for i in range(1):
        dataset = datasets.MNIST(args.data_path, train=False,
                                 transform=transform,
                                 download=True)
        loader = data_load.DataLoader(dataset, shuffle=False, batch_size=200)
        print("Loading model")
        # Define Network
        model = resnet.resnet18(pretrained=False)
        com_model = ComDefend.ComDefend()

        # Load pre-trained weights
        com_model.load_state_dict(torch.load(args.best_com_model_path))
        com_model.eval().to(device)
        model.load_state_dict(torch.load(args.best_cla_model_path))
        model.eval().to(device)
        # model = resnet.resnet18(pretrained=False)
        # pretrained_cla_dict = torch.load(source_model)
        # model_dict = model.state_dict()
        #
        # pretrained_cla_weight_key = []
        # pretrained_cla_weight_value = []
        # model_weight_key = []
        # model_weight_value = []
        #
        # for k, v in pretrained_cla_dict.items():
        #     # print(k)
        #     pretrained_cla_weight_key.append(k)
        #     pretrained_cla_weight_value.append(v)
        #
        # for k, v in model_dict.items():
        #     # print(k)
        #     model_weight_key.append(k)
        #     model_weight_value.append(v)
        # if len(pretrained_cla_weight_key) == len(model_weight_value):
        #     model.load_state_dict(torch.load(source_model))
        #     print("Weights Loaded!")
        #     model.to(device)
        # else:
        #     new_dict = {}
        #     for i in range(len(model_dict)):
        #         new_dict[model_weight_key[i]] = pretrained_cla_weight_value[i + 1]
        #     model_dict.update(new_dict)
        #     model.load_state_dict(model_dict)
        #     print("Weights Loaded!")
        #     model.to(device)
        # model.eval().to(device)
        # model = joint_model_2(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2])
        # model.load_state_dict(torch.load(joint_paths[7]))

        images_clean_ddn, labels_ddn, y_pred_ddn, images_adv_ddn, y_pred_ddn_adversarial = [], [], [], [], []
        y_correct_ddn, y_correct_ddn_adversarial = 0, 0

        i = 1
        print("Running DDN attack")
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            model.noise = nn.Parameter(data=torch.zeros([len(y), 1, 28, 28]), requires_grad=False)
            model.to(device)

            x = com_model(x)
            x = torch.from_numpy(x.cpu().data.numpy()).type(torch.FloatTensor).to(device)
            x = torch.clamp(x, 0, 1)

            output = model(x)
            _, prediction = torch.max(output.data, 1)
            requires_grad_(model, False)

            attacker = DDN(steps=100, device=device)
            start = time.time()
            ddn_atk = attacker.attack(model, x, labels=y, targeted=False)
            ddn_time = time.time() - start

            # Save images
            all_imgs = torch.cat((x[0:16], ddn_atk[0:16]))
            image_name = "ddn_attack_" + str(i) + ".png"
            save_image(all_imgs, "D:/python_workplace/resnet-AE/outputData/DDN/mnist/" + image_name, nrow=16, pad_value=0)

            # Print metrics
            pred_orig = model(x).argmax(dim=1)
            pred_ddn = model(ddn_atk).argmax(dim=1)

            y_correct_ddn += pred_orig.eq(y.data).cpu().sum().item()
            y_correct_ddn_adversarial += pred_ddn.eq(y.data).cpu().sum().item()

            images_clean_ddn.extend(list(x.cpu().data.numpy()))
            labels_ddn.extend(list(y.cpu().data.numpy()))
            y_pred_ddn.extend(list(pred_orig.cpu().data.numpy()))
            images_adv_ddn.extend(list(ddn_atk.cpu().data.numpy()))
            y_pred_ddn_adversarial.extend(list(pred_ddn.cpu().data.numpy()))

            print("Batch:[%d/%d]:" % (i, len(loader)))
            print("Predictions on Original Images: {}".format(pred_orig.cpu().data.numpy()))
            print("Predictions on DDN Attack: {}".format(pred_ddn.cpu().data.numpy()))
            print("DDN Done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.".format(
                ddn_time, (pred_ddn.cpu() != y.cpu()).float().mean().item() * 100, l2_norm(ddn_atk - x).mean().item()))

            i += 1

        total_images = len(dataset)
        correct_before = y_correct_ddn / total_images
        correct_after_ddn = y_correct_ddn_adversarial / total_images
        print("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_ddn))

        with open(args.output_ddn, "w+") as f:
            f.write("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_ddn))
            f.write("\n")

        with open(args.output_path_ddn_train, "wb") as f:
            adv_data_dict = {
                "images_clean": images_clean_ddn,
                "images_adv": images_adv_ddn,
                "labels": labels_ddn,
                "y_pred_test": y_pred_ddn,
                "y_pred_test_adversarial": y_pred_ddn_adversarial,
            }
            pickle.dump(adv_data_dict, f)

        # # Define Network
        # model = resnet.resnet18(pretrained=False)
        # com_model = ComDefend.ComDefend()
        #
        # # Load pre-trained weights
        # com_model.load_state_dict(torch.load(args.best_com_model_path))
        # com_model.to(device)
        # model.load_state_dict(torch.load(args.best_cla_model_path))
        # model.to(device)
        #
        # dataset = datasets.MNIST(args.data_path, train=False,
        #                          transform=transform,
        #                          download=True)
        # loader = data_load.DataLoader(dataset, shuffle=False, batch_size=200)
        #
        # images_clean_ddn, labels_ddn, y_pred_ddn, images_adv_ddn, y_pred_ddn_adversarial = [], [], [], [], []
        # y_correct_ddn, y_correct_ddn_adversarial = 0, 0
        #
        # i = 1
        # print("Running DDN attack")
        # for data in loader:
        #     x, y = data
        #     x = x.to(device)
        #     y = y.to(device)
        #
        #     # com_model.eval()
        #     # x = com_model(x)
        #     # x = torch.from_numpy(x.cpu().data.numpy()).type(torch.FloatTensor).to(device)
        #     # x = torch.clamp(x, 0, 1)
        #
        #     model.noise = nn.Parameter(data=torch.zeros([len(y), 1, 28, 28]), requires_grad=False)
        #     model.to(device)
        #
        #     output = model(x)
        #     _, prediction = torch.max(output.data, 1)
        #     requires_grad_(model, False)
        #
        #     attacker = DDN(steps=100, device=device)
        #     start = time.time()
        #     ddn_atk = attacker.attack(model, x, labels=y, targeted=False)
        #     ddn_time = time.time() - start
        #
        #     # Save images
        #     all_imgs = torch.cat((x[0:16], ddn_atk[0:16]))
        #     image_name = "ddn_attack_" + str(i) + ".png"
        #     save_image(all_imgs, "D:/python_workplace/resnet-AE/outputData/DDN/mnist/" + image_name, nrow=16,
        #                pad_value=0)
        #
        #     # Print metrics
        #     pred_orig = model(x).argmax(dim=1)
        #     pred_ddn = model(ddn_atk).argmax(dim=1)
        #
        #     y_correct_ddn += pred_orig.eq(y.data).cpu().sum().item()
        #     y_correct_ddn_adversarial += pred_ddn.eq(y.data).cpu().sum().item()
        #
        #     images_clean_ddn.extend(list(x.cpu().data.numpy()))
        #     labels_ddn.extend(list(y.cpu().data.numpy()))
        #     y_pred_ddn.extend(list(pred_orig.cpu().data.numpy()))
        #     images_adv_ddn.extend(list(ddn_atk.cpu().data.numpy()))
        #     y_pred_ddn_adversarial.extend(list(pred_ddn.cpu().data.numpy()))
        #
        #     print("Batch:[%d/%d]:" % (i, len(loader)))
        #     print("Predictions on Original Images: {}".format(pred_orig.cpu().data.numpy()))
        #     print("Predictions on DDN Attack: {}".format(pred_ddn.cpu().data.numpy()))
        #     print("DDN Done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.".format(
        #         ddn_time, (pred_ddn.cpu() != y.cpu()).float().mean().item() * 100, l2_norm(ddn_atk - x).mean().item()))
        #
        #     i += 1
        #
        # total_images = len(dataset)
        # correct_before = y_correct_ddn / total_images
        # correct_after_ddn = y_correct_ddn_adversarial / total_images
        # print("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_ddn))

        # with open(args.output_ddn, "w+") as f:
        #     f.write("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_ddn))
        #     f.write("\n")
        #
        # with open(args.output_path_ddn_test, "wb") as f:
        #     adv_data_dict = {
        #         "images_clean": images_clean_ddn,
        #         "images_adv": images_adv_ddn,
        #         "labels": labels_ddn,
        #         "y_pred_test": y_pred_ddn,
        #         "y_pred_test_adversarial": y_pred_ddn_adversarial,
        #     }
        #     pickle.dump(adv_data_dict, f)

        images_clean_cw, labels_cw, y_pred_cw, images_adv_cw, y_pred_cw_adversarial = [], [], [], [], []
        y_correct_cw, y_correct_cw_adversarial = 0, 0

        j = 1
        print("Running C&W attack")
        dataset = datasets.MNIST(args.data_path, train=False,
                                 transform=transform,
                                 download=True)
        loader = data_load.DataLoader(dataset, shuffle=False, batch_size=200)
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)

            com_model.eval()
            x = com_model(x)
            x = torch.from_numpy(x.cpu().data.numpy()).type(torch.FloatTensor).to(device)
            x = torch.clamp(x, 0, 1)

            # model.noise = nn.Parameter(data=torch.zeros([len(y), 1, 28, 28]), requires_grad=False)
            model.to(device)

            output = model(x)
            _, prediction = torch.max(output.data, 1)
            requires_grad_(model, False)

            cwattacker = CarliniWagnerL2(device=device, image_constraints=(0, 1), num_classes=10)
            start = time.time()
            cw_atk = cwattacker.attack(model, x, labels=y, targeted=False)
            cw_time = time.time() - start

            # Save images
            all_imgs = torch.cat((x[0:16], cw_atk[0:16]))
            image_name = "cw_attack_" + str(j) + ".png"
            save_image(all_imgs, "D:/python_workplace/resnet-AE/outputData/CW/mnist/" + image_name, nrow=16, pad_value=0)

            pred_orig = model(x).argmax(dim=1)
            pred_cw = model(cw_atk).argmax(dim=1)

            y_correct_cw += pred_orig.eq(y.data).cpu().sum().item()
            y_correct_cw_adversarial += pred_cw.eq(y.data).cpu().sum().item()

            images_clean_cw.extend(list(x.cpu().data.numpy()))
            labels_cw.extend(list(y.cpu().data.numpy()))
            y_pred_cw.extend(list(pred_orig.cpu().data.numpy()))
            images_adv_cw.extend(list(cw_atk.cpu().data.numpy()))
            y_pred_cw_adversarial.extend(list(pred_cw.cpu().data.numpy()))

            print("Batch:[%d/%d]:" % (j, len(loader)))
            print("Predictions on Original Images: {}".format(pred_orig.cpu().data.numpy()))
            print("Predictions on C&W Attack: {}".format(pred_cw.cpu().data.numpy()))
            print("C&W Done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.".format(
                cw_time, (pred_cw.cpu() != y.cpu()).float().mean().item() * 100, l2_norm(cw_atk - x).mean().item()))

            j += 1

        total_images = len(dataset)
        correct_before = y_correct_cw / total_images
        correct_after_cw = y_correct_cw_adversarial / total_images
        print("Test Set Accuracy Before: %.03f | Accuracy After CW: %.03f" % (correct_before, correct_after_cw))
        #
        # with open(args.output_cw, "w+") as f:
        #     f.write("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_cw))
        #     f.write("\n")
        #
        # with open(args.output_path_cw, "wb") as f:
        #     adv_data_dict = {
        #         "images_clean": images_clean_cw,
        #         "images_adv": images_adv_cw,
        #         "labels": labels_cw,
        #         "y_pred_test": y_pred_cw,
        #         "y_pred_test_adversarial": y_pred_cw_adversarial,
        #     }
        #     pickle.dump(adv_data_dict, f)
