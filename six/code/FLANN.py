#FLANN匹配，最邻近搜索的快速库，FLANN具有一种内部机制，该机制可以根据数据集本身选择最合适的算法来处理数据。
import cv2
from matplotlib import pyplot as plt
img_orign = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_orign.jpg', cv2.IMREAD_GRAYSCALE)
img_find = cv2.imread('E:\\Git\\opencv\\six\\resource\\picture_find.JPG', cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
kp_orign, des_orign = sift.detectAndCompute(img_orign, None)
kp_find, des_find = sift.detectAndCompute(img_find, None)
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks = 50)
#FLANN匹配器有两个参数：indexParams与searchParams。这两个参数以字典的形式传递
#indexParams：配置索引及其参数，索引可选项包括LinearIndex, KTreeIndex, KMeansIndex, CompositeIndex和AutotuneIndex
#这里选择KtreeIndex，KtreeIndex配置索引只需要指定待处理核密度树的数量， 最理想的数量在1 - 16之间，并且kd-trees可被并行处理
#searchParams字典只包含一个字段checks，用来指定索引树要被遍历的次数，该值越高，计算匹配花费的时间越长。
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des_find, des_orign, k=2)
#过滤满足匹配到最近像素点与次近像素点间的比例小于一定阈值的像素点，及保留特征较为突出的关键点
matchesMask = [[0, 0] for i in range(len(matches))]
for i,(m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
# drawParams = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0, 0, ), matchesMask = matchesMask, flags = 0)
# img_result = cv2.drawMatchesKnn(img_find, kp_find, img_orign, kp_orign, matches, None, **drawParams)
img_result = cv2.drawMatchesKnn(img_find, kp_find, img_orign, kp_orign, matches, img_orign, flags=2, matchesMask = matchesMask)
plt.imshow(img_result),plt.show()