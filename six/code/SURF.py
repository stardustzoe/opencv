#使用快速Hessian算法和SURF来提取和检测特征
#SURF采用Hessian算法检测关键点，而SURF会提取特征
#SURF也是一种与图像比例无关的斑点检测算法，速度优于SIFT
import cv2
import numpy as np
img = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_1.jpeg')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#参数为Hessian阀值，阈值越大能识别的特征越少。
sift = cv2.xfeatures2d.SURF_create(8000)
#keypoint对象（angle, class_id, octave, pt(像素坐标)，reponse, size）
#keypoints 关键点keypoint对象List
#descriptor 特征描述，二维矩阵
keypoints, descriptor = sift.detectAndCompute(gray, None)
paper = np.ones(img.shape) * 255
paper = cv2.drawKeypoints(image=img, outImage=paper, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (255, 0, 0))
cv2.imshow('paper', paper)
cv2.waitKey()
cv2.destroyAllWindows()