#ORB特征匹配
#1.向FAST增加一个快速准确的方向向量
#2.能高效计算带方向的BRIEF特征
#3.基于带方向的BRIEF的方差分析和相关性分析
#4.在旋转不变性条件下学习一种不相关的BRIEF特征，这会在最邻近的应用中得到较好的性能
#Brute-Force暴力匹配方法是一种描述符匹配方法，该方法会比较两个描述符，并产生匹配结果列表。
#该方法基本上不涉及优化，第一个描述符的所有特征都用来和第二个描述符的特征进行比较，
#每次比较都会给出一个距离值，而最好的匹配结果会被认为是一个匹配
import cv2
from matplotlib import pyplot as plt
img_orign = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_orign.jpg', cv2.IMREAD_GRAYSCALE)
img_find = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_find.JPG', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp_orign, des_orign = orb.detectAndCompute(img_orign, None)
kp_find, des_find = orb.detectAndCompute(img_find, None)
#暴力匹配模式
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
#match对象（distance, imgIdx, queryIdx, trainIdx）
#返回match的list
# matches = bf.match(des_find, des_orign)
#matches = sorted(matches, key = lambda x:x.distance)
#img_result = cv2.drawMatches(img_find, kp_find, img_orign, kp_orign, matches[:80], img_orign, flags=2)
#k-最邻近匹配模式
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
#返回的matchs中，每个元素是由k个matche对象组成的集合，k为匹配到最近的k个元素
matches = bf.knnMatch(des_find, des_orign, k=2)
#过滤满足匹配到最近像素点与次近像素点间的比例小于一定阈值的像素点，及保留特征较为突出的关键点
matchesMask = [[0, 0] for i in range(len(matches))]
for i,(m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
img_result = cv2.drawMatchesKnn(img_find, kp_find, img_orign, kp_orign, matches, img_orign, flags=2, matchesMask=matchesMask)

plt.imshow(img_result),plt.show()