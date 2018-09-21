#使用DOG于SIFT进行特征的提取于描述
#SIFT分别采用DOG和SIFT来检测关键点并提取关键点周围的特征
#Difference of Gaussians (DOG) 通过对同一图像使用不同的高斯滤波器来获得感兴趣的区域（关键点）
#SIFT算法会使用GOG检测关键点，并且对关键点周围的区域计算特征向量
#SIFT算法是一种与图像比例无关的斑点检测算法
import cv2
import numpy as np
img = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_1.jpeg')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
#keypoint对象（angle, class_id, octave, pt(像素坐标)，reponse, size）
#keypoints 关键点keypoint对象List
#descriptor 特征描述，二维矩阵
keypoints, descriptor = sift.detectAndCompute(gray, None)
paper = np.ones(img.shape) * 255
paper = cv2.drawKeypoints(image=img, outImage=paper, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (255, 0, 0))
cv2.imshow('paper', paper)
cv2.waitKey()
cv2.destroyAllWindows()