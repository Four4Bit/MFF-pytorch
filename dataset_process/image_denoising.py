import numpy as np
import cv2


# 单个图像去噪方法（光流图像为灰度图）
def single_grayscale_denoising(image_path):
    img = cv2.imread(image_path)
    # - h：决定滤波器强度的参数。较高的h值可以更好地消除噪点，但同时也可以消除图像细节。（可以设为10)
    # - hForColorComponents：与h相同，但仅用于彩色图像。（通常与h相同）
    # - templateWindowSize：应为奇数。（建议设为7）
    # - searchWindowSize：应为奇数。（建议设为21）
    dst = cv2.fastNlMeansDenoising(img, None, 3, 7, 21)
    return dst


# 图像平滑以达到降噪,均值模糊,ksize值越大越模糊
def jun_zhi(image_path, ksize):
    img = cv2.imread(image_path)
    # ksize*ksize的窗口大小的卷积核，应为奇数
    if ksize%2 != 1:
        print('ksize为卷积核边长，应为奇数')
        return
    dst = cv2.blur(img, (ksize, ksize))
    return dst


# 图像平滑以达到降噪,高斯模糊，同样的ksize比均值模糊程度低
def gao_si(image_path, ksize, sigmaX, sigmaY):
    img = cv2.imread(image_path)
    # sigmaX和sigmaY为横轴与纵轴权重的标准差，若为0则opencv根据内核大小自动推算方差大小
    if ksize % 2 != 1:
        print('ksize为卷积核边长，应为奇数')
        return
    dst = cv2.GaussianBlur(img, (ksize, ksize), sigmaX, sigmaY)
    return dst


# 图像平滑以达到降噪，中值模糊，ksize越大，过滤椒盐噪声效果越好，但图像越模糊
def zhong_zhi(image_path, ksize):
    img = cv2.imread(image_path)
    # ksize*ksize的窗口大小的卷积核，设为奇数
    if ksize % 2 != 1:
        print('ksize为卷积核边长，应为奇数')
        return
    dst = cv2.medianBlur(img, ksize)
    return dst


# 图像平滑以达到降噪，双边模糊
def shuang_bian(image_path, ksize, sigmaColor, sigmaSpace):
    img = cv2.imread(image_path)
    # sigmaColor是灰度差值权重的标准差，sigmaSpace是位置权重的标准差，这两个标准差越大滤波能力越强
    if ksize % 2 != 1:
        print('ksize为卷积核边长，应为奇数')
        return
    # sigmaColor/sigmaSpace, 21/21, 31/31, 41/41
    dst = cv2.bilateralFilter(img, ksize, sigmaColor, sigmaSpace)
    return dst


def main():
    # 源图片路径，精确到image文件
    single_image_path = r'D:\document\service outsourcing\quzaoTest\1\00001.jpg'
    # 目标路径
    tar_image_path = r'D:\document\service outsourcing\quzaoTest\output'
    # 下面五个函数参数值可自行设定
    img1 = single_grayscale_denoising(single_image_path)
    img2 = jun_zhi(single_image_path, 5)
    img3 = gao_si(single_image_path, 5, 0, 0)
    img4 = zhong_zhi(single_image_path, 5)
    img5 = shuang_bian(single_image_path, 5, 21, 21)

    cv2.imwrite(tar_image_path + '//denoising.jpg', img1)
    cv2.imwrite(tar_image_path + '//junzhi.jpg', img2)
    cv2.imwrite(tar_image_path + '//gaosi.jpg', img3)
    cv2.imwrite(tar_image_path + '//zhongzhi.jpg', img4)
    cv2.imwrite(tar_image_path + '//shuangbian.jpg', img5)


if __name__ == '__main__':
    main()
