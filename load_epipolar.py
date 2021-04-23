
import numpy as np
import glob
import cv2


    
def load_points():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #IMAGENES 1,3,4,6,7,12,13,
    files1=glob.glob('images\\left*.jpg')
    files2=glob.glob('images\\right*.jpg')
    img_points1=[]
    images1=[]
    images2=[]
    img_points2=[]
    n=0
    for i in range(len(files1)):
        fname1=files1[i]
        img1 = cv2.imread(fname1)
        
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        
        fname2=files2[i]
        img2 = cv2.imread(fname2)
        
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, (7,6),None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (7,6),None)
        # If found, add object points, image points (after refining them)
        if ret1 == True and ret2==True:       
            cornersf1 = cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
            cornersf2 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
            img1=img1[:,:,0]
            images1.append(img1)
            img2=img2[:,:,0]
            images2.append(img2)
            # image_plot(img1,np.transpose(np.array(cornersf1[:,0,:])))
            # image_plot(img2,np.transpose(np.array(cornersf2[:,0,:])))
            img_points1.append(cornersf1)
            img_points2.append(cornersf2)
            n+=1
    
    
    x_1=np.zeros((2,1))
    for i in img_points1:
        xi=np.transpose(np.array(i[:,0,:]))
        x_1=np.concatenate((x_1,xi),axis=1)
    x_1=x_1[:,1:] 
    x_2=np.zeros((2,1))
    for i in img_points2:
        xi=np.transpose(np.array(i[:,0,:]))
        x_2=np.concatenate((x_2,xi),axis=1)
    x_2=x_2[:,1:] 
    return x_1,x_2,images1,images2,n

