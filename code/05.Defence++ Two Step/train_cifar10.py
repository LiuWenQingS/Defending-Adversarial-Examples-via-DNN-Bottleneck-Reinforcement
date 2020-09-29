import cla_resnet_cifar10
import FGSM_resnet_cifar10
import reTrain_cifar10
import step_training_advSet_cifar10
import step_training_cleanSet_cifar10
import numpy as np
import torch
import random

if __name__ == "__main__":
    torch.cuda.manual_seed(10)
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    # train autoencoder with clean set
    # 10 rounds and 4 block exchanges
    # best_cla_epoch_f = cla_cifar10.train_cla()
    # for i in range(3, 4):
    #     best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_" \
    #                           + str(282) + ".pth"
    #     for j in range(5):
    #         FGSM_cifar10.FGSM(best_cla_model_path, j + 1, i + 1, "AE_CleanSet", "cuda:0")
    #         best_ae_epoch = ae_cleanSet_cifar10.train_ae(best_cla_model_path, j + 1, i + 1, "AE_CleanSet", "cuda:0")
    #         best_cla_epoch = reTrain_cifar10.reTrain(best_cla_model_path, best_ae_epoch, j + 1, i + 1, "AE_CleanSet", "cuda:0")
    #         best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/" \
    #                               "ResNet18/cifar10/block_" + str(i + 1) + "/round_" + str(j + 1) + \
    #                               "/model/model_" + str(best_cla_epoch) + ".pth"

    # train autoencoder with adv set
    # 10 rounds and 4 block exchanges
    for i in range(4):
        best_cla_model_path = "D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/cifar10/model/model_" \
                              + str(282) + ".pth"
        best_ae_model_path = ""
        for j in range(10):
            FGSM_cifar10.FGSM(best_cla_model_path, j + 1, i + 1, "AE_AdvSet", "cuda:0")
            best_ae_epoch = ae_advSet_cifar10.train_ae(best_cla_model_path, best_ae_model_path, j + 1, i + 1, "AE_AdvSet", "cuda:0")
            best_ae_model_path = "H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/Autoencoder/" \
                                  "ResNet18/cifar10/block_" + str(i + 1) + "/round_" + str(j + 1) + \
                                  "/model/model_" + str(best_ae_epoch) + ".pth"
            best_cla_epoch = reTrain_cifar10.reTrain(best_cla_model_path, best_ae_epoch, j + 1, i + 1, "AE_AdvSet", "cuda:0")
            best_cla_model_path = "H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/" \
                                  "ResNet18/cifar10/block_" + str(i + 1) + "/round_" + str(j + 1) + \
                                  "/model/model_" + str(best_cla_epoch) + ".pth"