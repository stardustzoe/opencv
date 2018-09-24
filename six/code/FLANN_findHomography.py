#FLANN匹配的单应性
#单应性是一个条件，该条件表明当两幅图片中的一副出现投影畸变时，他们还能彼此匹配
import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img_orign = cv2.imread('E:\\Git\\opencv\\six\\resource\\projection_orign.jpg', 0)
img_find = cv2.imread('E:\\Git\\opencv\\six\\resource\\projection_find.jpg', 0)
sift = cv2.xfeatures2d.SIFT_create()
kp_orign, des_orign = sift.detectAndCompute(img_orign, None)
kp_find, des_find = sift.detectAndCompute(img_find, None)
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks = 50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des_find, des_orign, k = 2)
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
# 设置只有存在10个以上匹配时，才会提取两幅图像中匹配点的坐标，并通过findHomography函数，获取两幅图像的透视变换矩阵H,
# 然后使用perspectiveTransform函数对原图像的四个顶点使用变换矩阵H, 获得其变换后的坐标
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_orign[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_find[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #找到两个平面之间的透视变换。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5, 0)
    matchesMask = mask.ravel().tolist()
    h, w = img_orign.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    #使用变换矩阵计算变换后的坐标
    dst = cv2.perspectiveTransform(pts, M)
    img_find = cv2.polylines(img_find, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
drawParams = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
img_result = cv2.drawMatches(img_find, kp_find, img_orign, kp_orign, good, None, **drawParams)
plt.imshow(img_result),plt.show()