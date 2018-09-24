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

# IMG1	第一个源图像。
# keypoints1	来自第一个源图像的关键点。
# IMG2	第二个源图像。
# keypoints2	来自第二个源图像的关键点。
# matches1to2	匹配从第一个图像到第二个图像，这意味着keypoints1 [i]在关键点2 [匹配[i]]中具有对应点。
# outImg	输出图像。其内容取决于定义输出图像中绘制内容的标志值。请参阅下面的可能标志位值。
# MatchColor	匹配的颜色（线和连接的关键点）。如果matchColor == Scalar :: all（-1），则颜色随机生成。
# singlePointColor	单个关键点（圆圈）的颜色，这意味着关键点没有匹配项。如果singlePointColor == Scalar :: all（-1），则颜色随机生成。
# matchesMask	掩码确定绘制了哪些匹配。如果掩码为空，则绘制所有匹配项。
# flag	标志设置绘图功能。可能的标志位值由DrawMatchesFlags定义。
# 枚举  	{
#   DEFAULT = 0，将创建输出图像矩阵（Mat :: create），即可以重用输出图像的现有存储器。将绘制两个源图像，匹配和单个关键点。对于每个关键点，仅绘制中心点（没有关键点大小和方向的关键点周围的圆圈）。
#   DRAW_OVER_OUTIMG = 1，不会创建输出图像矩阵（Mat :: create）。将在输出图像的现有内容上绘制匹配。
#   NOT_DRAW_SINGLE_POINTS = 2，不会绘制单个关键点。
#   DRAW_RICH_KEYPOINTS = 4 对于每个关键点，将绘制具有关键点大小和方向的关键点周围的圆。
# }
img_result = cv2.drawMatchesKnn(img_find, kp_find, img_orign, kp_orign, matches, img_orign, flags=2, matchesMask=matchesMask)
plt.imshow(img_result),plt.show()