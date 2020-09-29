import argparse
import torch
import time
import foolbox
import pickle
from torch import nn
from models import resnet_cifar
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from fast_adversarial.fast_adv.models.mnist import SmallCNN
from fast_adversarial.fast_adv.attacks import DDN, CarliniWagnerL2
from fast_adversarial.fast_adv.utils import requires_grad_, l2_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate adversarial examples on CIFAR10")
    parser.add_argument("--data_path", default="D:/python_workplace/resnet-AE/inputData/cifar/cifar10/cifar-10-batches-py/")
    parser.add_argument("--model_path",
                        default="D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_282.pth")
    parser.add_argument("--output_path_cw", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/CW/cifar10/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_path_ddn", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/cifar10/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_path_ddn_train", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/cifar10/train/train.pkl",
                        help="Output directory with test images.")

    args = parser.parse_args()

    # model_path = "D:/python_workplace/resnet-AE/checkpoint/AdversarialLearning/ResNet18/cifar10/model/model_50.pth"
    # model_path = "D:/python_workplace/resnet-AE/checkpoint/Joint_Training/ResNet18/cifar10/block_1/decoder_4_6/model/model_100.pth"

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Loading data")
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR10(args.data_path, train=True,
                               transform=transform,
                               download=True)
    loader = data.DataLoader(dataset, shuffle=False, batch_size=500)

    print("Loading model")
    model = resnet_cifar.resnet18(pretrained=False)
    # model_dict = model.state_dict()
    # pretrained_ae_model = torch.load(args.model_path)
    # model_key = []
    # model_value = []
    # pretrained_ae_key = []
    # pretrained_ae_value = []
    # for k, v in model_dict.items():
    #     # print(k)
    #     model_key.append(k)
    #     model_value.append(v)
    # for k, v in pretrained_ae_model.items():
    #     # print(k)
    #     pretrained_ae_key.append(k)
    #     pretrained_ae_value.append(v)
    # new_dict = {}

    # print(pretrained_ae_key[175:])
    # for i in range(len(model_dict)):
    #     if i < 30:
    #         new_dict[model_key[i]] = pretrained_ae_value[i]
    #     else:
    #         new_dict[model_key[i]] = pretrained_ae_value[i + 145]
    #
    # for i in range(len(model_dict)):
    #     if i < 30:
    #         new_dict[model_key[i]] = pretrained_ae_value[i]
    #     else:
    #         new_dict[model_key[i]] = pretrained_ae_value[i + 85]

    # for i in range(len(model_dict)):
    #     if i < 30:
    #         new_dict[model_key[i]] = pretrained_ae_value[i]
    #     else:
    #         new_dict[model_key[i]] = pretrained_ae_value[i + 205]
    #
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # model.to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(device)

    images_clean_cw, labels_cw, y_pred_cw, images_adv_cw, y_pred_cw_adversarial = [], [], [], [], []
    y_correct_cw, y_correct_cw_adversarial = 0, 0

    images_clean_ddn, labels_ddn, y_pred_ddn, images_adv_ddn, y_pred_ddn_adversarial = [], [], [], [], []
    y_correct_ddn, y_correct_ddn_adversarial = 0, 0

    i = 1
    print("Running DDN attack")
    for data in loader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        model.to(device)

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
        save_image(all_imgs, "D:/python_workplace/resnet-AE/outputData/DDN/cifar10/" + image_name, nrow=16, pad_value=0)

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

    with open(args.output_path_ddn_train, "wb") as f:
        adv_data_dict = {
            "images_clean": images_clean_ddn,
            "images_adv": images_adv_ddn,
            "labels": labels_ddn,
            "y_pred_test": y_pred_ddn,
            "y_pred_test_adversarial": y_pred_ddn_adversarial,
        }
        pickle.dump(adv_data_dict, f)

    # j = 1
    # print("Running C&W attack")
    # for data in loader:
    #     x, y = data
    #     x = x.to(device)
    #     y = y.to(device)
    #     model.to(device)
    #
    #     output = model(x)
    #     _, prediction = torch.max(output.data, 1)
    #     requires_grad_(model, False)
    #
    #     cwattacker = CarliniWagnerL2(device=device, image_constraints=(0, 1), num_classes=10)
    #     start = time.time()
    #     cw_atk = cwattacker.attack(model, x, labels=y, targeted=False)
    #     cw_time = time.time() - start
    #
    #     # Save images
    #     all_imgs = torch.cat((x[0:16], cw_atk[0:16]))
    #     image_name = "cw_attack_" + str(j) + ".png"
    #     save_image(all_imgs, "D:/python_workplace/resnet-AE/outputData/CW/cifar10/" + image_name, nrow=16, pad_value=0)
    #
    #     pred_orig = model(x).argmax(dim=1)
    #     pred_cw = model(cw_atk).argmax(dim=1)
    #
    #     y_correct_cw += pred_orig.eq(y.data).cpu().sum().item()
    #     y_correct_cw_adversarial += pred_cw.eq(y.data).cpu().sum().item()
    #
    #     images_clean_cw.extend(list(x.cpu().data.numpy()))
    #     labels_cw.extend(list(y.cpu().data.numpy()))
    #     y_pred_cw.extend(list(pred_orig.cpu().data.numpy()))
    #     images_adv_cw.extend(list(cw_atk.cpu().data.numpy()))
    #     y_pred_cw_adversarial.extend(list(pred_cw.cpu().data.numpy()))
    #
    #     print("Batch:[%d/%d]:" % (j, len(loader)))
    #     print("Predictions on Original Images: {}".format(pred_orig.cpu().data.numpy()))
    #     print("Predictions on C&W Attack: {}".format(pred_cw.cpu().data.numpy()))
    #     print("C&W Done in {:.1f}s: Success: {:.2f}%, Mean L2: {:.4f}.".format(
    #         cw_time, (pred_cw.cpu() != y.cpu()).float().mean().item() * 100, l2_norm(cw_atk - x).mean().item()))
    #
    #     j += 1
    #
    # total_images = len(dataset)
    # correct_before = y_correct_cw / total_images
    # correct_after_cw = y_correct_cw_adversarial / total_images
    # print("Test Set Accuracy Before: %.03f | Accuracy After CW: %.03f" % (correct_before, correct_after_cw))
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
