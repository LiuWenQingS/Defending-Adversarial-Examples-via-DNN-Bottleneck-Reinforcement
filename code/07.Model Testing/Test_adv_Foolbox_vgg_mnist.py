import argparse
import torch
import time
import foolbox
import pickle
from torch import nn
from models import resnet
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from fast_adversarial.fast_adv.models.mnist import SmallCNN
from fast_adversarial.fast_adv.attacks import DDN, CarliniWagnerL2
from fast_adversarial.fast_adv.utils import requires_grad_, l2_norm
from models.vgg import vgg16_bn


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate adversarial examples on MNIST")
    parser.add_argument("--data_path", default="D:/python_workplace/resnet-AE/inputData/mnist/")
    parser.add_argument("--model_path",
                        default="D:/python_workplace/resnet-AE/checkpoint/Classification/VGG/mnist/model/model_150.pth")
    parser.add_argument("--output_path_cw", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/CW/vgg/mnist/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_path_ddn", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/vgg/mnist/test/test.pkl",
                        help="Output directory with test images.")
    parser.add_argument("--output_ddn", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/DDN/vgg/mnist/acc.txt",
                        help="Output directory with test images.")
    parser.add_argument("--output_cw", type=str,
                        default="D:/python_workplace/resnet-AE/outputData/CW/vgg/mnist/acc.txt",
                        help="Output directory with test images.")

    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Loading data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.MNIST(args.data_path, train=False,
                             transform=transform,
                             download=True)
    loader = data.DataLoader(dataset, shuffle=False, batch_size=1000)

    for i in range(1):
        print("Loading model")
        model = vgg16_bn(pretrained=False)
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
            model.noise = nn.Parameter(data=torch.zeros([len(y), 1, 28, 28]), requires_grad=False)
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

        with open(args.output_path_ddn, "wb") as f:
            adv_data_dict = {
                "images_clean": images_clean_ddn,
                "images_adv": images_adv_ddn,
                "labels": labels_ddn,
                "y_pred_test": y_pred_ddn,
                "y_pred_test_adversarial": y_pred_ddn_adversarial,
            }
            pickle.dump(adv_data_dict, f)

    j = 1
    print("Running C&W attack")
    for data in loader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        model.noise = nn.Parameter(data=torch.zeros([len(y), 1, 28, 28]), requires_grad=False)
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

    with open(args.output_cw, "w+") as f:
        f.write("Test Set Accuracy Before: %.03f | Accuracy After DDN: %.03f" % (correct_before, correct_after_cw))
        f.write("\n")

    with open(args.output_path_cw, "wb") as f:
        adv_data_dict = {
            "images_clean": images_clean_cw,
            "images_adv": images_adv_cw,
            "labels": labels_cw,
            "y_pred_test": y_pred_cw,
            "y_pred_test_adversarial": y_pred_cw_adversarial,
        }
        pickle.dump(adv_data_dict, f)
