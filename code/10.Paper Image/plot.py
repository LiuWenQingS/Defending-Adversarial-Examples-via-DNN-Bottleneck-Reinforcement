import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np
# from torchvision import models
import os


def plot_test_1_Model_Robustness():
    with open("D:/python_workplace/resnet-AE/test/Model_Robustness/acc_Source Model.txt", "r") as f1:
        acc_source_model = f1.readlines()
        epsilon = []
        for i in range(len(acc_source_model)):
            acc_source_model[i] = acc_source_model[i].strip()
            epsilon.append(float(acc_source_model[i].split("|")[0].split(":")[-1]))
            acc_source_model[i] = float(acc_source_model[i].split(":")[-1].split("%")[0])
            # print(acc_source_model[i])
        # print(epsilon)

    with open("D:/python_workplace/resnet-AE/test/Model_Robustness/acc_Adv FGSM Model.txt", "r") as f2:
        acc_adv_FGSM = f2.readlines()
        for i in range(len(acc_adv_FGSM)):
            acc_adv_FGSM[i] = acc_adv_FGSM[i].strip()
            acc_adv_FGSM[i] = float(acc_adv_FGSM[i].split(":")[-1].split("%")[0])
            # print(acc_adv_FGSM[i])

    with open("D:/python_workplace/resnet-AE/test/Model_Robustness/acc_Joint Model.txt", "r") as f3:
        acc_ETUDE = f3.readlines()
        for i in range(len(acc_ETUDE)):
            acc_ETUDE[i] = acc_ETUDE[i].strip()
            acc_ETUDE[i] = float(acc_ETUDE[i].split(":")[-1].split("%")[0])
            # print(acc_ETUDE[i])

    plt.figure(figsize=(44, 32))
    plt.plot(epsilon, acc_source_model,  color='g', label="No defense", linewidth=10)
    # plt.plot(epsilon, acc_adv_FGSM, label="Adv.FGSM Model")
    plt.plot(epsilon, acc_ETUDE, color='r', label="Defend++", linewidth=10)
    plt.tick_params(labelsize=80)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 200}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 150}
    plt.xlabel("Distortion(ε)", font1)
    plt.ylabel("Adv. Acc", font1)
    # plt.title("Models' Accuracy On Adversarial Examples")

    plt.legend(prop=font2)

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/Model_Robustness/magnitude.png")
    plt.show()


def plot_test_2_ToyModel_Robustness_FCSize():
    path_cla = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/Classification/"
    size_catalog = os.listdir(path_cla)
    size = [int(i.split("_")[-1]) for i in size_catalog]
    acc_cla = []
    for i in size_catalog:
        with open(path_cla + i + "/" + "best_acc.txt") as f1:
            acc_cla.append(float(f1.readline().strip().split("=")[-1].split("%")[0]))
    # print(acc_cla)
    size_new = np.linspace(100, 900, 1000)
    acc_new = spline(size, acc_cla, size_new)
    plt.figure(figsize=(8, 6))
    plt.plot(size_new, acc_new)
    # plt.plot(size, acc_cla)
    plt.xlabel("FC Size")
    plt.ylabel("Accuracy(%)")
    plt.title("MNIST Test Set Accuracy With Different FC Size")

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_1.png")
    plt.show()

    path_re = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/AE_CleanSet/ReTrain/"
    path_fgsm = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/AE_CleanSet/"
    rounds_catalog = os.listdir(path_re + size_catalog[0] + "/")
    rounds = [int(i.split("_")[-1]) for i in rounds_catalog]
    size_acc = [[], [], [], [], [], [], [], [], []]
    fgsm_acc = [[], [], [], [], [], [], [], [], []]
    for i in range(len(size_catalog)):
        for j in range(len(rounds_catalog)):
            with open(path_re + size_catalog[i] + "/" + rounds_catalog[j] + "/" + "best_acc.txt") as f2:
                size_acc[i].append(float(f2.readline().strip().split("=")[-1].split("%")[0]))
            with open(path_fgsm + size_catalog[i] + "/" + rounds_catalog[j] + "/" + "acc.txt") as f3:
                fgsm_acc[i].append(float(f3.readlines()[-1].strip().split(":")[-1].split("%")[0]))
    print(size_acc)
    print(fgsm_acc)

    plt.figure(figsize=(8, 6))
    for i in range(len(size)):
        plt.plot(rounds, size_acc[i], label="FC Size %d" % (size[i]))
    # plt.plot(size, acc_cla)
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With Different FC Size")
    plt.legend()

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_2.png")
    plt.show()

    fig1 = plt.figure(figsize=(20, 18))
    fig1.suptitle("Adversarial Examples Accuracy Before and After Training")

    plt.subplot(331)
    plt.plot(rounds, size_acc[0], label="After Training")
    plt.plot(rounds, fgsm_acc[0], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 100")
    plt.legend()

    plt.subplot(332)
    plt.plot(rounds, size_acc[1], label="After Training")
    plt.plot(rounds, fgsm_acc[1], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 200")
    plt.legend()

    plt.subplot(333)
    plt.plot(rounds, size_acc[2], label="After Training")
    plt.plot(rounds, fgsm_acc[2], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 300")
    plt.legend()

    plt.subplot(334)
    plt.plot(rounds, size_acc[3], label="After Training")
    plt.plot(rounds, fgsm_acc[3], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 400")
    plt.legend()

    plt.subplot(335)
    plt.plot(rounds, size_acc[4], label="After Training")
    plt.plot(rounds, fgsm_acc[4], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 500")
    plt.legend()

    plt.subplot(336)
    plt.plot(rounds, size_acc[5], label="After Training")
    plt.plot(rounds, fgsm_acc[5], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 600")
    plt.legend()

    plt.subplot(337)
    plt.plot(rounds, size_acc[6], label="After Training")
    plt.plot(rounds, fgsm_acc[6], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 700")
    plt.legend()

    plt.subplot(338)
    plt.plot(rounds, size_acc[7], label="After Training")
    plt.plot(rounds, fgsm_acc[7], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 800")
    plt.legend()

    plt.subplot(339)
    plt.plot(rounds, size_acc[8], label="After Training")
    plt.plot(rounds, fgsm_acc[8], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 900")
    plt.legend()

    # fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.4)

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_3.png")
    plt.show()

    max_acc = [max(i) for i in size_acc]
    # print(max_acc)

    size_new = np.linspace(100, 900, 1000)
    acc_new = spline(size, max_acc, size_new)
    plt.figure(figsize=(8, 6))
    plt.plot(size_new, acc_new)
    # plt.plot(size, acc_cla)
    plt.xlabel("FC Size")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy in Best Model Trained With Different FC Size AE")

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_4.png")
    plt.show()

    path_re = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/AE_AdvSet/ReTrain/"
    path_fgsm = "D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/FGSM/AE_AdvSet/"
    rounds_catalog = os.listdir(path_re + size_catalog[0] + "/")
    rounds = [int(i.split("_")[-1]) for i in rounds_catalog]
    size_acc = [[], [], [], [], [], [], [], [], []]
    fgsm_acc = [[], [], [], [], [], [], [], [], []]
    for i in range(len(size_catalog)):
        for j in range(len(rounds_catalog)):
            with open(path_re + size_catalog[i] + "/" + rounds_catalog[j] + "/" + "best_acc.txt") as f2:
                size_acc[i].append(float(f2.readline().strip().split("=")[-1].split("%")[0]))
            with open(path_fgsm + size_catalog[i] + "/" + rounds_catalog[j] + "/" + "acc.txt") as f3:
                fgsm_acc[i].append(float(f3.readlines()[-1].strip().split(":")[-1].split("%")[0]))
    print(size_acc)
    print(fgsm_acc)

    plt.figure(figsize=(8, 6))
    for i in range(len(size)):
        plt.plot(rounds, size_acc[i], label="FC Size %d" % (size[i]))
    # plt.plot(size, acc_cla)
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With Different FC Size")
    plt.legend()

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_5.png")
    plt.show()

    fig1 = plt.figure(figsize=(20, 18))
    fig1.suptitle("Adversarial Examples Accuracy Before and After Training")

    plt.subplot(331)
    plt.plot(rounds, size_acc[0], label="After Training")
    plt.plot(rounds, fgsm_acc[0], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 100")
    plt.legend()

    plt.subplot(332)
    plt.plot(rounds, size_acc[1], label="After Training")
    plt.plot(rounds, fgsm_acc[1], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 200")
    plt.legend()

    plt.subplot(333)
    plt.plot(rounds, size_acc[2], label="After Training")
    plt.plot(rounds, fgsm_acc[2], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 300")
    plt.legend()

    plt.subplot(334)
    plt.plot(rounds, size_acc[3], label="After Training")
    plt.plot(rounds, fgsm_acc[3], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 400")
    plt.legend()

    plt.subplot(335)
    plt.plot(rounds, size_acc[4], label="After Training")
    plt.plot(rounds, fgsm_acc[4], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 500")
    plt.legend()

    plt.subplot(336)
    plt.plot(rounds, size_acc[5], label="After Training")
    plt.plot(rounds, fgsm_acc[5], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 600")
    plt.legend()

    plt.subplot(337)
    plt.plot(rounds, size_acc[6], label="After Training")
    plt.plot(rounds, fgsm_acc[6], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 700")
    plt.legend()

    plt.subplot(338)
    plt.plot(rounds, size_acc[7], label="After Training")
    plt.plot(rounds, fgsm_acc[7], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 800")
    plt.legend()

    plt.subplot(339)
    plt.plot(rounds, size_acc[8], label="After Training")
    plt.plot(rounds, fgsm_acc[8], label="Before Training")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy With FC Size 900")
    plt.legend()

    # fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.4)

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_6.png")
    plt.show()

    max_acc = [max(i) for i in size_acc]
    # print(max_acc)

    size_new = np.linspace(100, 900, 1000)
    acc_new = spline(size, max_acc, size_new)
    plt.figure(figsize=(8, 6))
    plt.plot(size_new, acc_new)
    # plt.plot(size, acc_cla)
    plt.xlabel("FC Size")
    plt.ylabel("Accuracy(%)")
    plt.title("Adversarial Examples Accuracy in Best Model Trained With Different FC Size AE")

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/ToyModel_Robustness_FCSize/image_7.png")
    plt.show()


def plot_test_3_AdVSet_CleanAE_Dor_Dar():
    with open("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/log.txt", "r") as f1:
        log = f1.readlines()
        id = []
        distance_or = []
        distance_ar = []
        Dor_divide_Dar = []
        for i in range(len(log)):
            log[i] = log[i].strip()
            id.append(int(log[i].split(",")[0].split(":")[-1]))
            distance_or.append(float(log[i].split(",")[1].split("=")[-1]))
            distance_ar.append(float(log[i].split(",")[2].split("=")[-1]))
            Dor_divide_Dar.append(float(log[i].split(",")[3].split("=")[-1]))
        # print(Dor_divide_Dar)
        d_0_1, d_1_2, d_2_3, d_3_4, d_4_5, d_5_6, d_6_7, d_7_8, d_8_9, d_9_10 = [], [], [], [], [], [], [], [], [], []
        for j in Dor_divide_Dar:
            if 0 < j <= 0.1:
                d_0_1.append(j)
            elif 0.1 < j <= 0.2:
                d_1_2.append(j)
            elif 0.2 < j <= 0.3:
                d_2_3.append(j)
            elif 0.3 < j <= 0.4:
                d_3_4.append(j)
            elif 0.4 < j <= 0.5:
                d_4_5.append(j)
            elif 0.5 < j <= 0.6:
                d_5_6.append(j)
            elif 0.6 < j <= 0.7:
                d_6_7.append(j)
            elif 0.7 < j <= 0.8:
                d_7_8.append(j)
            elif 0.8 < j <= 0.9:
                d_8_9.append(j)
            elif 0.9 < j <= 1:
                d_9_10.append(j)
            else:
                print("There is a error")
        print("0-0.1: %d/%.2f%% | 0.1-0.2: %d/%.2f%% | 0.2-0.3: %d/%.2f%% | 0.3-0.4: %d/%.2f%% | " %
              (len(d_0_1), len(d_0_1) / 10000 * 100, len(d_1_2), len(d_1_2) / 10000 * 100, len(d_2_3),
               len(d_2_3) / 10000 * 100, len(d_3_4), len(d_3_4) / 10000 * 100))
        print("0.4-0.5: %d/%.2f%% | 0.5-0.6: %d/%.2f%% | 0.6-0.7: %d/%.2f%% | 0.7-0.8: %d/%.2f%% | " %
              (len(d_4_5), len(d_4_5) / 10000 * 100, len(d_5_6), len(d_5_6) / 10000 * 100, len(d_6_7),
               len(d_6_7) / 10000 * 100, len(d_7_8), len(d_7_8) / 10000 * 100))
        print("0.8-0.9: %d/%.2f%% | 0.9-1.0: %d/%.2f%% |" %
              (len(d_8_9), len(d_8_9) / 10000 * 100, len(d_9_10), len(d_9_10) / 10000 * 100))
        # print(len(d_0_1) + len(d_1_2) + len(d_2_3) + len(d_3_4) + len(d_4_5) + len(d_5_6) + len(d_6_7) + len(d_7_8))

        # # figure 1
        # fig1 = plt.figure(figsize=(20, 10))
        # fig1.suptitle("Distance OR and AR")
        # plt.subplot(221)
        # plt.scatter(id, distance_or)
        #
        # plt.xlabel("Image ID")
        # plt.ylabel("Distance")
        # plt.title("MSE of Original and Reconstructed Image in Clean Set Trained Model")
        #
        # plt.subplot(222)
        # plt.hist(distance_or, 100)
        #
        # plt.xlabel("Distance")
        # plt.ylabel("Frequency")
        # plt.xlim(0, 0.1, 0.01)
        # plt.title("MSE of Original and Reconstructed Image Distribution Map in Clean Set Trained Model")
        #
        # plt.subplot(223)
        # plt.scatter(id, distance_ar)
        #
        # plt.xlabel("Image ID")
        # plt.ylabel("Distance")
        # plt.title("MSE of Adversarial Examples and Reconstructed Image in Clean Set Trained Model")
        #
        # plt.subplot(224)
        # plt.hist(distance_ar, 100)
        #
        # plt.xlabel("Distance")
        # plt.ylabel("Frequency")
        # plt.xlim(0, 0.1, 0.01)
        # plt.title("MSE of Adversarial Examples and Reconstructed Image Distribution Map in Clean Set Trained Model")
        #
        # img = plt.gcf()
        # img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/image_5.png")
        # plt.show()
        #
        # # figure 2
        # fig1 = plt.figure(figsize=(20, 6))
        # fig1.suptitle("OR Divide AR")
        # plt.subplot(121)
        # plt.scatter(id, Dor_divide_Dar)
        #
        # plt.xlabel("Image ID")
        # plt.ylabel("Distance")
        # plt.title("Distance OR Divide Distance AR Scatter Map in Clean Set Trained Model")
        # # plt.legend()
        #
        # plt.subplot(122)
        # plt.hist(Dor_divide_Dar, 100)
        #
        # plt.xlabel("Distance")
        # plt.ylabel("Frequency")
        # plt.xlim(0, 1, 0.05)
        # plt.title("Distance OR Divide Distance AR Distribution Map in Clean Set Trained Model")
        # # plt.legend()
        #
        # img = plt.gcf()
        # img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/image_6.png")
        # plt.show()

        # figure 3
        plt.figure(figsize=(44, 32))
        divide = plt.hist(Dor_divide_Dar, 100, color='r')
        plt.tick_params(labelsize=100)
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 200}
        plt.xlabel("R", font1)
        plt.ylabel("Frequency", font1)
        x_r, y_r = [], []
        for i in range(100):
            if divide[0][i] != 0:
                x_r.append(divide[1][i])
                y_r.append(divide[0][i])
        # plt.plot(x_r, y_r, color='r')
        img = plt.gcf()
        img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/distance.pdf")
        plt.show()

        # # figure 3
        # # plt.figure(figsize=(10, 10))
        # d_or = plt.hist(distance_or, 100, label="Distance OR")
        # d_ar = plt.hist(distance_ar, 100, label="Distance AR")
        # # plt.xlim(0, 0.1, 0.01)
        # # plt.ylim(0, 1400, 200)
        # plt.xlabel("Distance")
        # plt.ylabel("Frequency")
        # plt.title("Distance OR and AR")
        # x_or, y_or, x_ar, y_ar = [], [], [], []
        # for i in range(100):
        #     if d_or[0][i] != 0:
        #         x_or.append(d_or[1][i])
        #         y_or.append(d_or[0][i])
        #     if d_ar[0][i] != 0:
        #         x_ar.append(d_ar[1][i])
        #         y_ar.append(d_ar[0][i])
        #
        # plt.legend()
        # img = plt.gcf()
        # img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/image_8.png")
        # plt.show()
        #
        # # plt.figure(figsize=(10, 10))
        # plt.plot(x_or, y_or, label="Distance OR")
        # plt.plot(x_ar, y_ar, label="Distance AR")
        # plt.xlabel("Distance")
        # plt.ylabel("Frequency")
        # plt.title("Distance OR and AR")
        # plt.legend()
        #
        # img = plt.gcf()
        # img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/image_7.png")
        # plt.show()


def plot_test_4_Suboptimal_AE_Model():
    path_sub_clean_ae_s = "H:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/Autoencoder/ResNet18/mnist"
    path_sub_clean_re_s = "H:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/ResNet18/mnist"
    path_sub_adv_ae_s = "H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/Autoencoder/ResNet18/mnist"
    path_sub_adv_re_s = "H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/mnist"
    path_best_clean_ae_s = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/Autoencoder/ResNet18/mnist"
    path_best_clean_re_s = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/ResNet18/mnist"
    path_best_adv_ae_s = "D:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/Autoencoder/ResNet18/mnist"
    path_best_adv_re_s = "D:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/ResNet18/mnist"
    path_sub = [path_sub_clean_re_s, path_sub_adv_re_s]
    path_best = [path_best_clean_re_s, path_best_adv_re_s]
    path = [path_sub, path_best]
    blocks = os.listdir(path_sub_clean_ae_s)
    rounds = os.listdir(path_sub_clean_ae_s + "/" + blocks[0])
    rounds.append(rounds.pop(1))
    # print(rounds)
    sub_clean, sub_adv, best_clean, best_adv = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
    acc_sub = [sub_clean, sub_adv]
    acc_best = [best_clean, best_adv]
    acc = [acc_sub, acc_best]
    best_epoch = [[], []]

    for i in range(len(path)):
        for j in range(len(path[i])):
            for k in range(len(blocks)):
                for m in range(5 * (i + 1)):
                   with open(path[i][j] + "/" + blocks[k] + "/" + rounds[m] + "/" + "best_acc.txt") as f:
                       file = f.readline()
                       # print(file)
                       acc[i][j][k].append(float(file.strip().split("=")[-1].split("%")[0]))
                       best_epoch[i].append(int(float(file.strip().split(",")[0].split("=")[-1])))

    # print(sub_clean)
    # print(best_clean)
    clean_num, adv_num = 0, 0
    for i in best_epoch[1][0:40]:
        if i <= 50:
            clean_num += 1
    for i in best_epoch[1][40:80]:
        if i <= 50:
            adv_num += 1
    print("Total Best Results for Clean Trained Models: %d | Epoch of Best Result Less Than 20: %d" %
          (len(best_epoch[1][0:40]), clean_num))
    print("Total Best Results for Adv Trained Models: %d | Epoch of Best Result Less Than 20: %d" %
          (len(best_epoch[1][40:80]), adv_num))

    round_sub = [i + 1 for i in range(5)]
    round_best = [i + 1 for i in range(10)]

    # round_sub_new = np.linspace(1, 5, 300)
    # sub_clean_1 = spline(round_sub, sub_clean[0], round_sub_new)
    # plt.plot(round_sub_new, sub_clean_1, label="Block 1")
    # sub_clean_2 = spline(round_sub, sub_clean[1], round_sub_new)
    # plt.plot(round_sub_new, sub_clean_2, label="Block 2")
    # sub_clean_3 = spline(round_sub, sub_clean[2], round_sub_new)
    # plt.plot(round_sub_new, sub_clean_3, label="Block 3")
    # sub_clean_4 = spline(round_sub, sub_clean[3], round_sub_new)
    # plt.plot(round_sub_new, sub_clean_4, label="Block 4")
    # plt.show()

    # figure 1
    fig1 = plt.figure(figsize=(18, 10))
    fig1.suptitle("Accuracy of Adversarial Examples")
    plt.subplot(221)
    plt.plot(round_sub, sub_clean[0], label="Block 1")
    plt.plot(round_sub, sub_clean[1], label="Block 2")
    plt.plot(round_sub, sub_clean[2], label="Block 3")
    plt.plot(round_sub, sub_clean[3], label="Block 4")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Suboptimal Clean Set Trained AE Model")
    plt.legend()

    plt.subplot(222)
    plt.plot(round_sub, sub_adv[0], label="Block 1")
    plt.plot(round_sub, sub_adv[1], label="Block 2")
    plt.plot(round_sub, sub_adv[2], label="Block 3")
    plt.plot(round_sub, sub_adv[3], label="Block 4")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Suboptimal Adv Set Trained AE Model")
    plt.legend()

    plt.subplot(223)
    plt.plot(round_best[0:5], best_clean[0][0:5], label="Block 1")
    plt.plot(round_best[0:5], best_clean[1][0:5], label="Block 2")
    plt.plot(round_best[0:5], best_clean[2][0:5], label="Block 3")
    plt.plot(round_best[0:5], best_clean[3][0:5], label="Block 4")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Best Clean Set Trained AE Model")
    plt.legend()

    plt.subplot(224)
    plt.plot(round_best[0:5], best_adv[0][0:5], label="Block 1")
    plt.plot(round_best[0:5], best_adv[1][0:5], label="Block 2")
    plt.plot(round_best[0:5], best_adv[2][0:5], label="Block 3")
    plt.plot(round_best[0:5], best_adv[3][0:5], label="Block 4")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Best Adv Set Trained AE Model")
    plt.legend()

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/Suboptimal_AE_Model/image_1.png")
    plt.show()

    # figure 2
    fig2 = plt.figure(figsize=(18, 10))
    fig2.suptitle("Accuracy of Adversarial Examples With Clean Set Trained AE Model")
    plt.subplot(221)
    plt.plot(round_sub, sub_clean[0], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_clean[0][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 1 Exchanged")
    plt.legend()

    plt.subplot(222)
    plt.plot(round_sub, sub_clean[1], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_clean[1][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 2 Exchanged")
    plt.legend()

    plt.subplot(223)
    plt.plot(round_sub, sub_clean[2], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_clean[2][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 3 Exchanged")
    plt.legend()

    plt.subplot(224)
    plt.plot(round_sub, sub_clean[3], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_clean[3][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 4 Exchanged")
    plt.legend()

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/Suboptimal_AE_Model/image_2.png")
    plt.show()

    # figure 3
    fig3 = plt.figure(figsize=(18, 10))
    fig3.suptitle("Accuracy of Adversarial Examples With Adv Set Trained AE Model")
    plt.subplot(221)
    plt.plot(round_sub, sub_adv[0], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_adv[0][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 1 Exchanged")
    plt.legend()

    plt.subplot(222)
    plt.plot(round_sub, sub_adv[1], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_adv[1][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 2 Exchanged")
    plt.legend()

    plt.subplot(223)
    plt.plot(round_sub, sub_adv[2], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_adv[2][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 3 Exchanged")
    plt.legend()

    plt.subplot(224)
    plt.plot(round_sub, sub_adv[3], label="Suboptimal Model")
    plt.plot(round_best[0:5], best_adv[3][0:5], label="Best Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy of Adversarial Examples With Block 4 Exchanged")
    plt.legend()

    img = plt.gcf()
    img.savefig("D:/python_workplace/resnet-AE/test/Suboptimal_AE_Model/image_3.png")
    plt.show()


if __name__ == "__main__":
    plot_test_1_Model_Robustness()
    # plot_test_2_ToyModel_Robustness_FCSize()
    # plot_test_3_AdVSet_CleanAE_Dor_Dar()
    # plot_test_4_Suboptimal_AE_Model()
