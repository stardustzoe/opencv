#特征检测算法cornerHarris，用于检测角点
#角点在图像旋转的情况下也能被检测到
#图像比例会影响检测
#该方法共有4个参数分别为img,blockSize,ksize-Sobel,k-Harris
#img - 数据类型为float32的输入图像
#blockSize - 角点检测中要考虑的领域大小
#ksize-Sobel - Sobel算子的中孔，必须为3和31之间的奇数
#ksize-Harris - 角点检测方法中的自由参数，取值范围为[0.04， 0.06]
import cv2
import numpy as np
img = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_1.jpeg')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
#dst尺寸与gray一致，其中每个像素的值为 R = det(M) - k (trace(M))^2,值越大表示该像素点为角点的可能性越大
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
paper = np.ones(img.shape) * 255
#取大于dst最大值*0.01的像素点，并设置其颜色为蓝色
paper[dst > 0.01 * dst.max()] = [255, 0, 0]
cv2.imshow('paper', paper)
cv2.waitKey()
cv2.destroyAllWindows()

