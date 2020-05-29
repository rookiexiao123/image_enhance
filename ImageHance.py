import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dir = '/home/xhz/object_detaction/keras-yolo3-master/VOCdevkit/VOC2020/tree'

for image in os.listdir(dir):
    #1.resize
    img = cv2.imread(dir + '/' + image)
    img = cv2.resize(img, (416, 416))
    img = cv2.imwrite(dir + '/' + image, img)

    img = Image.open(dir + '/' + image)
    img = np.array(img)


    #2.左右翻转，保存图片
    flipped_img = np.fliplr(img)
    cv2.imwrite(dir + '/' + 'spin' + image, flipped_img)


    HEIGHT = img.shape[0] #图像的垂直尺寸（高度）
    WIDTH = img.shape[1] #图像的水平尺寸（宽度）
    DEPTH = img.shape[2] #图像的通道数

    #3.左平移,保存图片
    left_img = np.array(range(0, HEIGHT * WIDTH * DEPTH)).reshape(HEIGHT, WIDTH, DEPTH)

    for i in range(HEIGHT, 1, -1):
        for j in range(WIDTH):
            if(i < HEIGHT -20):
                left_img[j][i] = img[j][i - 20]
            elif (i < HEIGHT - 1):
                left_img[j][i] = 0
    cv2.imwrite(dir + '/' + 'left' + image, left_img)

    #4.右平移，保存图片
    right_img = np.array(range(0, HEIGHT * WIDTH * DEPTH)).reshape(HEIGHT, WIDTH, DEPTH)

    for j in range(WIDTH):
        for i in range(HEIGHT):
            if(i < HEIGHT -20):
                right_img[j][i] = img[j][i + 20]

    cv2.imwrite(dir + '/' + 'right' + image, right_img)

    #5.向上平移
    top_img = np.array(range(0, HEIGHT * WIDTH * DEPTH)).reshape(HEIGHT, WIDTH, DEPTH)

    for j in range(WIDTH):
        for i in range(HEIGHT):
            if(j < WIDTH - 20 and j > 20):
                top_img[j][i] = img[j + 20][i]
            else:
                top_img[j][i] = 0

    cv2.imwrite(dir + '/' + 'top' + image, top_img)

    #6.向下平移
    down_img = np.array(range(0, HEIGHT * WIDTH * DEPTH)).reshape(HEIGHT, WIDTH, DEPTH)
    for j in range(WIDTH, 1, -1):
        for i in range(WIDTH):
            if(j < WIDTH-20 and j > 20):
                down_img[j][i] = img[j - 20][i]

    cv2.imwrite(dir + '/' + 'down' + image, down_img)

    #7.加上噪声
    noise_img = np.array(range(0, HEIGHT * WIDTH * DEPTH)).reshape(HEIGHT, WIDTH, DEPTH)
    noise = np.random.randint(5, size = (HEIGHT, WIDTH, 4))

    for i in range(WIDTH):
        for j in range(HEIGHT):
            for k in range(DEPTH):
                if (img[i][j][k] != 255):
                    noise_img[i][j][k] = img[i][j][k]
                    noise_img[i][j][k] += noise[i][j][k]
    cv2.imwrite(dir + '/' + 'noise' + image, noise_img)
