
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from copy import copy
def main_load():
    files1=glob.glob('images\\diffleft.jpg')
    files2=glob.glob('images\\diffright.jpg')
    images1=[]
    images2=[]
    img_points1=[]
    img_points2=[]
    n_puntos=0
    for i in range(len(files1)):
         x_1,x_2,F,img2,img1=opencv_epipolar(files1[i],files2[i])   
         if len(x_1) and len(x_2):
             n_puntos=len(x_1[0,:])
             img_points1.append(x_1)
             img_points2.append(x_2)
             images1.append(img1)
             images2.append(img2)
    
    x_1=np.zeros((2,1))
    for i in img_points1:
        x_1=np.concatenate((x_1,i),axis=1)
    x_1=x_1[:,1:] 
    x_2=np.zeros((2,1))
    for i in img_points2:
        x_2=np.concatenate((x_2,i),axis=1)
    x_2=x_2[:,1:] 
    return x_1,x_2,images1,images2,n_puntos,F


def opencv_epipolar(filename1,filename2):
    img1 = cv2.imread(filename1,0)  #queryimage # left image
    img2 = cv2.imread(filename2,0) #trainimage # right image
    image1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    image2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    x_1=np.transpose(pts1)
    x_2=np.transpose(pts2)

    
    # fig,ax=plt.subplots()
    # ax.plot(pts1[:,0],pts1[:,1],'b.')
    # ax.plot(pts2[:,0],pts2[:,1],'r.')
    # ax.set_ylim(480,0)
    #plt.show(block=False)
    
    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    #plt.imshow(img5)
    # plt.subplot(121),plt.imshow(img5)
    # plt.subplot(122),plt.imshow(img3)
    fig=plt.figure()
    plt.imshow(img3)
    plt.show()
    

    return x_1,x_2,F,image2,image1