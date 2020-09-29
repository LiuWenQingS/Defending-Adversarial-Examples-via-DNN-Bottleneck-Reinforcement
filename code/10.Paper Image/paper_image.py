import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave


root1 = "D:/picture/1/source.png"
root2 = "D:/picture/2/source.png"
root3 = "D:/python_workplace/resnet-AE/inputData/mnist/Image-mnist/train/6/66_6.png"

img = cv2.imread(root2)
img1 = img.astype('float')[:, :, 0]

img_dct = cv2.dct(img1)  # 进行离散余弦变换
# img_dct_log = np.log(abs(img_dct))  # 进行log处理
# for i in range(500):
#     for j in range(500):
#         if i < 62 and j < 62:
#             continue
#         else:
#             img_dct[i, j] = 0
for i in range(500):
    for j in range(500):
        if i < 250 and j < 250:
            img_dct[i, j] = 0
        else:
            continue

# for i in range(500):
#     for j in range(500):
#         if img_dct_log[i, j] < 0:
#             img_dct_log[i, j] = 0
#         if img_dct_log[i, j] > 255:
#             img_dct_log[i, j] = 255

# for i in range(500):
#     for j in range(500):
#         if i < 125 and j < 125:
#             img_dct_log[i, j] = 10.826283996300345
#         if i < 250 and j < 250:
#             continue
#         else:
#             img_dct_log[i, j] = 10.826283996300345

# mask = np.zeros((500, 500))
# mask[crow - r:crow + r, ccol - r:ccol + r] = 1
# mask_img = res1 * mask
# mask1 = np.zeros((500, 500))
# mask1[:, :] = 255
# mask1[crow - r:crow + r, ccol - r:ccol + r] = 0
# mask_img = mask_img + mask1
# print(mask_img)

# for i in range(500):
#     for j in range(500):
#         if img_dct_log[i, j] > 5:
#             img_dct_log[i, j] = 1
#         else:
#             img_dct_log[i, j] = 0

# max_num = img_dct_log[0, 0]
# min_num = img_dct_log[0, 0]
# for i in range(500):
#     for j in range(500):
#         if img_dct_log[i, j] > max_num:
#             max_num = img_dct_log[i, j]
#         if img_dct_log[i, j] < min_num:
#             min_num = img_dct_log[i, j]
#
# print(max_num)
#
# for i in range(500):
#     for j in range(500):
#         if img_dct_log[i, j] != 0:
#             img_dct_log[i, j] = (max_num - img_dct_log[i, j]) / (max_num - min_num)

# print(img_dct_log)

# plt.imshow(img_dct_log, "gray")

# img_idct = cv2.idct(img_dct)
img_idct = cv2.idct(img_dct)  # 进行离散余弦反变换

# img_dct_log = cv2.normalize(img_dct_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# cv2.imwrite("D:/picture/2/4-3.png", img_dct_log)
# imsave("D:/picture/2/保留左上4分之1低频信息DCT变换.png", img_idct)
# plt.imshow(img1, 'gray')
# plt.title('original image')
#
# plt.imshow(img_dct_log)
# plt.title('DCT2(cv2_dct)')
#

cv2.imwrite("D:/picture/Figure-5/1.png", img_idct)
plt.imshow(img_idct, 'gray')
plt.title('IDCT2(cv2_idct)')

plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # 读取图像
# figure = cv2.imread(root2)
# row, col, channel = figure.shape
# img1, img2, img3 = np.zeros([row, col]), np.zeros([row, col]), np.zeros([row, col])
# img1, img2, img3 = figure[:, :, 0], figure[:, :, 1], figure[:, :, 2]
# # print(img == img2)
#
# # 傅里叶变换
# dct1 = cv2.dct(np.float32(img1), flags=cv2.DCT_INVERSE)
# # fshift1 = np.fft.fftshift(dft1)
#
# dct2 = cv2.dct(np.float32(img2), flags=cv2.DCT_INVERSE)
# # fshift2 = np.fft.fftshift(dft2)
#
# dct3 = cv2.dct(np.float32(img3), flags=cv2.DCT_INVERSE)
# # fshift3 = np.fft.fftshift(dft3)
#
# # 频谱图像低通图
# # res1 = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
# fimg1 = 20 * np.log(cv2.magnitude(dct1[:, :, 0], dct1[:, :, 1]))
# fimg2 = 20 * np.log(cv2.magnitude(dct2[:, :, 0], dct2[:, :, 1]))
# fimg3 = 20 * np.log(cv2.magnitude(dct3[:, :, 0], dct3[:, :, 1]))
# # rows, cols = res1.shape
# # crow, ccol = int(rows / 2), int(cols / 2) # 中心位置
# # mask = np.zeros((rows, cols))
# # r = 31
# # mask[crow - r:crow + r, ccol - r:ccol + r] = 1
# # mask_img = res1 * mask
# # mask1 = np.zeros((rows, cols))
# # mask1[:, :] = 255
# # mask1[crow - r:crow + r, ccol - r:ccol + r] = 0
# # mask_img = mask_img + mask1
# # # print(mask_img)
# # cv2.imwrite("D:/picture/2/1.png", mask_img)
#
# # 频谱图像高通图
# # res1= 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
# # rows, cols = res1.shape
# # crow, ccol = int(rows / 2), int(cols / 2) # 中心位置
# #
# # mask = np.ones((rows, cols), np.uint8)
# # r = 125
# # mask[crow - r:crow + r, ccol - r:ccol + r] = 0
# # mask_img = res1 * mask
# # mask1 = np.zeros((rows, cols))
# # mask1[crow - r:crow + r, ccol - r:ccol + r] = 255
# # mask_img = mask_img + mask1
#
# # mask = np.zeros((rows, cols), np.uint8)
# # r = 62
# # mask[crow - r:crow + r, ccol - r:ccol + r] = 1
# # mask1 = np.ones((rows, cols), np.uint8)
# # r1 = 31
# # mask1[crow - r1:crow + r1, ccol - r1:ccol + r1] = 0
# # mask = mask * mask1
# # mask_img = res1 * mask
# # mask2 = np.zeros((rows, cols))
# # mask2[:, :] = 255
# # mask2[crow - r:crow + r, ccol - r:ccol + r] = 0
# # mask2[crow - r1:crow + r1, ccol - r1:ccol + r1] = 255
# # mask_img = mask_img + mask2
#
# # cv2.imwrite("D:/picture/2/1.png", mask_img)
#
# # 设置低通滤波器
# rows, cols = img1.shape
# crow, ccol = int(rows/2), int(cols/2)  # 中心位置
# mask = np.zeros((rows, cols, 2), np.uint8)
# r = 22
# mask[crow - r:crow + r, ccol - r:ccol + r] = 1
#
# f1 = fshift1 * mask
# f2 = fshift2 * mask
# f3 = fshift3 * mask
#
# f_new_1, f_new_2, f_new_3 = np.zeros((r * 2, r * 2, 2)), np.zeros((r * 2, r * 2, 2)), np.zeros((r * 2, r * 2, 2))
# f_new_1[:, :, :] = f1[crow - r:crow + r, ccol - r:ccol + r, :]
# f_new_2[:, :, :] = f2[crow - r:crow + r, ccol - r:ccol + r, :]
# f_new_3[:, :, :] = f3[crow - r:crow + r, ccol - r:ccol + r, :]
# f1, f2, f3 = f_new_1, f_new_2, f_new_3
#
# # 设置高通滤波器
# # rows, cols = img.shape
# # crow, ccol = int(rows/2), int(cols/2) #中心位置
# # mask = np.zeros((rows, cols, 2), np.uint8)
# # r = 62
# # mask[crow - r:crow + r, ccol - r:ccol + r] = 1
# # f = fshift * mask
# # mask = np.ones((rows, cols, 2), np.uint8)
# # r1 = 31
# # mask[crow - r1:crow + r1, ccol - r1:ccol + r1] = 0
# # f = f * mask
#
# # print(img.shape, f.shape, fshift.shape, mask.shape)
#
# # 傅里叶逆变换
# ishift1 = np.fft.ifftshift(f1)
# iimg1 = cv2.idft(ishift1)
# res1 = cv2.magnitude(iimg1[:, :, 0], iimg1[:, :, 1])
#
# ishift2 = np.fft.ifftshift(f2)
# iimg2 = cv2.idft(ishift2)
# res2 = cv2.magnitude(iimg2[:, :, 0], iimg2[:, :, 1])
#
# ishift3 = np.fft.ifftshift(f3)
# iimg3 = cv2.idft(ishift3)
# res3 = cv2.magnitude(iimg3[:, :, 0], iimg3[:, :, 1])
#
# # 总图
# # print(figure.shape, img1.shape, fimg1.shape, res1.shape)
# times = 8
# img, fimg, res = np.zeros([row, col, 3]), np.zeros([row, col, 3]), np.zeros([int(row/times)-1, int(col/times)-1, 3])
# img[:, :, 0], img[:, :, 1], img[:, :, 2] = img1, img2, img3
# fimg[:, :, 0], fimg[:, :, 1], fimg[:, :, 2] = fimg1, fimg2, fimg3
# max_num1, max_num2, max_num3 = 0, 0, 0
# min_num1, min_num2, min_num3 = res1[0, 0], res2[0, 0], res3[0, 0]
# for i in range(int(col/times)-1):
#     if max_num1 < max(res1[:, i]):
#         max_num1 = max(res1[:, i])
#     if max_num2 < max(res2[:, i]):
#         max_num2 = max(res2[:, i])
#     if max_num3 < max(res3[:, i]):
#         max_num3 = max(res3[:, i])
#     if min_num1 > min(res1[:, i]):
#         min_num1 = min(res1[:, i])
#     if min_num2 > min(res2[:, i]):
#         min_num2 = min(res2[:, i])
#     if min_num3 > min(res3[:, i]):
#         min_num3 = min(res3[:, i])
# for i in range(int(row/times)-1):
#     for j in range(int(col/times)-1):
#         res1[i, j] = int(255 * (res1[i, j] - min_num1) / (max_num1 - min_num1))
#         res2[i, j] = int(255 * (res2[i, j] - min_num2) / (max_num2 - min_num2))
#         res3[i, j] = int(255 * (res3[i, j] - min_num3) / (max_num3 - min_num3))
# res[:, :, 0], res[:, :, 1], res[:, :, 2] = res1, res2, res3
# img, fimg, res = img.astype("int"), fimg.astype("int"), res.astype("int")
#
# # 显示图像
# cv2.imwrite("D:/picture/1/1_1.png", res)
# plt.subplot(431), plt.imshow(img1, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(432), plt.imshow(fimg1, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(433), plt.imshow(res1, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
#
# plt.subplot(434), plt.imshow(img2, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(435), plt.imshow(fimg2, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(436), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
#
# plt.subplot(437), plt.imshow(img3, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(438), plt.imshow(fimg3, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(439), plt.imshow(res3, 'gray'), plt.title('Inverse Fourier Image')
#
# plt.subplot(4, 3, 10), plt.imshow(img), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(4, 3, 11), plt.imshow(fimg), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(4, 3, 12), plt.imshow(res), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()