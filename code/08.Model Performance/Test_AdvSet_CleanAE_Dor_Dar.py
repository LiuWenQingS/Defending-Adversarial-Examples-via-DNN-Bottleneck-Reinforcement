import torch
from torch import nn
from torchsummary import summary
import argparse
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from AutoEncoder import ResNet_AutoEncoder, BasicBlock_decoder, BasicBlock_encoder


def load_images(input_dir, batch_shape_clean, batch_shape_adv):
    image_clean = np.zeros(batch_shape_clean)
    image_adv = np.zeros(batch_shape_adv)
    idx = 0
    batch_size = batch_shape_adv[0]

    with open(input_dir, "rb") as f:
        data_dict = pickle.load(f)

    images_clean_list = data_dict["images_clean"]
    images_adv_list = data_dict["images_adv"]
    # print(len(images_adv_list))

    for j in range(len(images_clean_list)):
        image_clean[idx, 0, :, :] = images_clean_list[j]
        image_adv[idx, 0, :, :] = images_adv_list[j]

        idx += 1
        if idx == batch_size:
            yield idx, image_clean, image_adv
            image_clean = np.zeros(batch_shape_clean)
            image_adv = np.zeros(batch_shape_adv)
            idx = 0

    if idx > 0:
        yield idx, image_clean, image_adv


def dor_dar(best_ae_model_path, block, round, ae_training_set):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser("Adversarial Examples")
    parser.add_argument('--input_dir_trainSet', type=str,
                        default='D:/python_workplace/resnet-AE/outputData/FGSM/mnist/' + ae_training_set +
                                '/block_' + str(block) + '/round_' + str(round) + '/train/train.pkl',
                        help='data set dir path')
    parser.add_argument('--input_dir_testSet', type=str,
                        default='D:/python_workplace/resnet-AE/outputData/FGSM/mnist/' + ae_training_set +
                                '/block_' + str(block) + '/round_' + str(round) + '/test/test.pkl',
                        help='data set dir path')
    parser.add_argument('--image_size', type=int, default=28, help='Size of each input images.')
    parser.add_argument('--batch_size', type=int, default=1, help='How many images process at one time.')
    parser.add_argument('--lr', type=float, default=0.01, help='learing_rate. Default=0.01')
    parser.add_argument('--num_classes', type=int, default=10, help='num classes')
    parser.add_argument('--log_file_path', type=str,
                        default='D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/log.txt',
                        help='Save log file')

    args = parser.parse_args()

    model = ResNet_AutoEncoder(BasicBlock_encoder, BasicBlock_decoder, [2, 2, 2, 2]).to(device)
    # summary(model, (1, 28, 28))
    # print(model)

    # Load pre-trained weights
    pretrained_dict = torch.load(best_ae_model_path)
    model_dict = model.state_dict()
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # criterion
    criterion = nn.MSELoss()

    batchShape_clean = [args.batch_size, 1, args.image_size, args.image_size]
    batchShape_adv = [args.batch_size, 1, args.image_size, args.image_size]

    # initialize figure
    f, a = plt.subplots(3, 5, figsize=(12, 6))
    plt.ion()  # continuously plot

    with open(args.log_file_path, "w") as f1:
        print("Waiting for Testing!")
        with torch.no_grad():
            # loss_or_list = []
            # loss_ar_list = []
            sum_loss_or = 0
            sum_loss_ar = 0
            id = 1
            for batchSize, images_clean, images_adv in load_images(args.input_dir_testSet, batchShape_clean,
                                                                   batchShape_adv):
                model.eval()
                images_clean_tensor = torch.from_numpy(images_clean).type(torch.FloatTensor).to(device)
                images_adv_tensor = torch.from_numpy(images_adv).type(torch.FloatTensor).to(device)
                # print(images_clean)
                outputs = model(images_adv_tensor)
                loss_or = criterion(outputs, images_clean_tensor)
                # loss_or_list.append(loss_or.cpu().data.numpy())
                sum_loss_or += loss_or.cpu().data.numpy()

                loss_ar = criterion(outputs, images_adv_tensor)
                # loss_ar_list.append(loss_ar.cpu().data.numpy())
                sum_loss_ar += loss_ar.cpu().data.numpy()

                # if id == 1:
                #     # 观察原图以及去噪后的图片
                #     # first row:clean images
                #     for i in range(5):
                #         a[0][i].clear()
                #         a[0][i].imshow(np.reshape(images_clean[i, 0, :, :], (28, 28)), cmap='gray')
                #         a[0][i].set_xticks(())
                #         a[0][i].set_yticks(())
                #         a[0][i].set_title("Clean Images" + "[" + str(i + 1) + "]")
                #     # second row:adversarial images
                #     for i in range(5):
                #         a[1][i].clear()
                #         a[1][i].imshow(np.reshape(images_adv[i, 0, :, :], (28, 28)), cmap='gray')
                #         a[1][i].set_xticks(())
                #         a[1][i].set_yticks(())
                #         a[1][i].set_title("Adversarial Images" + "[" + str(i + 1) + "]")
                #     # third row:decoded images
                #     for i in range(5):
                #         a[2][i].clear()
                #         a[2][i].imshow(np.reshape(outputs.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                #         a[2][i].set_xticks(())
                #         a[2][i].set_yticks(())
                #         a[2][i].set_title("Decoded Images" + "[" + str(i + 1) + "]")
                #     plt.suptitle("Comparison Images")
                #     plt.draw()
                #     plt.pause(0.05)
                #     img = plt.gcf()
                #     img.savefig("D:/python_workplace/resnet-AE/test/AdVSet_CleanAE_Dor_Dar/image_7.png")
                #     plt.show()

                # 保存测试集loss至loss.txt文件中
                f1.write("ImageID: %d, Distance OR = %.3f, Distance AR = %.3f, Dor/Dar = %.3f" %
                         (id, loss_or, loss_ar, loss_or/loss_ar))
                f1.write('\n')
                f1.flush()
                id += 1
            print('Test Set OR Loss：%.4f' % sum_loss_or)
            print('Test Set AR Loss：%.4f' % sum_loss_ar)


if __name__ == "__main__":
    path_ae = "D:/python_workplace/resnet-AE/checkpoint/AE_CleanSet/Autoencoder/ResNet18/mnist/block_4/round_4/model/model_93.pth"
    block = 1
    round = 1
    traing_set = "AE_AdvSet"
    temp = [path_ae, block, round, traing_set]
    dor_dar(temp[0], temp[1], temp[2], temp[3])
