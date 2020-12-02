import cv2
import os
import numpy as np


# 原图像文件地址
img_path = "Games_img"

# 最后存放进行处理的图片文件
#img_convert_path = "image_convert"
img_convert_path = "Games_img"


'''读取文件中的图片，进行各种变换'''
filelist = os.listdir(img_path)
img_counts = 0



''' 运动模糊--- degree越大，模糊程度越高'''
def motion_blur(image, degree=8, angle=44):
    image = np.array(image)

    # 生成任意角度的运动模糊kernel的矩阵
    M = cv2.getRotationMatrix2D((degree , degree ), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred



for file in filelist:
    img_counts = img_counts +1
    f = os.path.join(img_path, file)
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    # 灰度化处理
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化图片
    #cv2.threshold(gray, 140, 255, 0, gray)  # 二值化函数
    # 模糊化
    #img_ = motion_blur(img)

    '''高斯模糊'''
    #img_ = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0.01, sigmaY=0.01)
    '''
    kernel_size = (5, 5);
    sigma = 0.5;
    img = cv2.GaussianBlur(img, kernel_size, sigma);
    '''
    ''' 进行旋转变换并且进行了缩放'''


    rows, cols, channels = img.shape
    M = cv2.getRotationMatrix2D((rows/2 , cols/2 ), 90, 0.8)
    img_1 = cv2.warpAffine(img,M, (cols, rows),borderValue=125)
    # 在平移一下，到中间
    pingyi = np.float32([[1,0,300],[0,1,0]])
    img_ = cv2.warpAffine(img_1, pingyi, (cols, rows), borderValue=125)

    # 保存图片
    #cv2.imwrite(os.path.join(img_convert_path, file + "_to_Gray"  + ".jpg"), gray)

    cv2.imwrite(os.path.join(img_convert_path, file + "_to_Rotate_90" + ".jpg"), img_)

print("There are ",img_counts," pictures")
