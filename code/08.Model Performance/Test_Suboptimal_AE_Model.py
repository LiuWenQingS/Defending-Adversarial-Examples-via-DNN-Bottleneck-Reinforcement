import cla_mnist
import FGSM_mnist
import reTrain_mnist
import ae_advSet_mnist
import ae_cleanSet_mnist

if __name__ == '__main__':
    # train autoencoder with clean set
    # 10 rounds and 4 block exchanges
    # best_cla_epoch_f = cla_mnist.train_cla()
    for i in range(4):
        best_cla_model_path = 'D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_' \
                              + str(66) + '.pth'
        for j in range(5):
            FGSM_mnist.FGSM(best_cla_model_path, j + 1, i + 1, 'AE_CleanSet')
            best_ae_epoch = ae_cleanSet_mnist.train_ae(best_cla_model_path, j + 1, i + 1, 'AE_CleanSet')
            best_cla_epoch = reTrain_mnist.reTrain(best_cla_model_path, best_ae_epoch, j + 1, i + 1, 'AE_CleanSet')
            best_cla_model_path = 'H:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/RetrainClassification/' \
                                  'ResNet18/mnist/block_' + str(i + 1) + '/round_' + str(j + 1) + \
                                  '/model/model_' + str(best_cla_epoch) + '.pth'

    # train autoencoder with adv set
    # 10 rounds and 4 block exchanges
    for i in range(4):
        best_cla_model_path = 'D:/python_workplace/resnet-AE/checkpoint/Classification/ResNet18/mnist/model/model_' \
                              + str(66) + '.pth'
        for j in range(5):
            FGSM_mnist.FGSM(best_cla_model_path, j + 1, i + 1, 'AE_AdvSet')
            best_ae_epoch = ae_advSet_mnist.train_ae(best_cla_model_path, j + 1, i + 1, 'AE_AdvSet')
            best_cla_epoch = reTrain_mnist.reTrain(best_cla_model_path, best_ae_epoch, j + 1, i + 1, 'AE_AdvSet')
            best_cla_model_path = 'H:/python_workplace/resnet-AE/checkpoint/AE_AdvSet/RetrainClassification/' \
                                  'ResNet18/mnist/block_' + str(i + 1) + '/round_' + str(j + 1) + \
                                  '/model/model_' + str(best_cla_epoch) + '.pth'
