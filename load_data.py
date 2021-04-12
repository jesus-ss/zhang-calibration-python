
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_model():
    model=np.loadtxt('Model1.txt')
    
    data1=np.loadtxt('data11.txt')
    data2=np.loadtxt('data21.txt')
    data3=np.loadtxt('data31.txt')
    data4=np.loadtxt('data41.txt')
    data5=np.loadtxt('data51.txt')
    

    X=np.array(list([model[:,i] for i in range(0,len(model[1,:]),2)]))
    Y=np.array(list([model[:,i] for i in range(1,len(model[1,:]),2)]))
    npoints=len(X)
    X_f=np.array([])
    Y_f=np.array([])
    for i in X:
        X_f=np.concatenate((X_f,i),axis=0)
        
    for i in Y:
        Y_f=np.concatenate((Y_f,i),axis=0)
    X_1=np.vstack((X_f.reshape(1,-1),Y_f.reshape(1,-1),np.ones(len(X_f))))
    x11=X_1
    X_2=X_1
    X_3=X_1
    X_4=X_1
    X_5=X_1
    
    
    x=np.array(list([data1[:,i] for i in range(0,len(data1[1,:]),2)]))
    y=np.array(list([data1[:,i] for i in range(1,len(data1[1,:]),2)]))
    x_f=np.array([])
    y_f=np.array([])
    for i in x:
        x_f=np.concatenate((x_f,i),axis=0)
        
    for i in y:
        y_f=np.concatenate((y_f,i),axis=0)
        
    x_1=np.vstack((x_f,y_f))
    
    
    x=np.array(list([data2[:,i] for i in range(0,len(data2[1,:]),2)]))
    y=np.array(list([data2[:,i] for i in range(1,len(data2[1,:]),2)]))
    x_f=np.array([])
    y_f=np.array([])
    for i in x:
        x_f=np.concatenate((x_f,i),axis=0)
        
    for i in y:
        y_f=np.concatenate((y_f,i),axis=0)
        
    x_2=np.vstack((x_f,y_f))
    
    x=np.array(list([data3[:,i] for i in range(0,len(data3[1,:]),2)]))
    y=np.array(list([data3[:,i] for i in range(1,len(data3[1,:]),2)]))
    x_f=np.array([])
    y_f=np.array([])
    for i in x:
        x_f=np.concatenate((x_f,i),axis=0)
        
    for i in y:
        y_f=np.concatenate((y_f,i),axis=0)
        
    x_3=np.vstack((x_f,y_f))
    
    
    x=np.array(list([data4[:,i] for i in range(0,len(data4[1,:]),2)]))
    y=np.array(list([data4[:,i] for i in range(1,len(data4[1,:]),2)]))
    x_f=np.array([])
    y_f=np.array([])
    for i in x:
        x_f=np.concatenate((x_f,i),axis=0)
        
    for i in y:
        y_f=np.concatenate((y_f,i),axis=0)
        
    x_4=np.vstack((x_f,y_f))
    
    x=np.array(list([data5[:,i] for i in range(0,len(data5[1,:]),2)]))
    y=np.array(list([data5[:,i] for i in range(1,len(data5[1,:]),2)]))
    x_f=np.array([])
    y_f=np.array([])
    for i in x:
        x_f=np.concatenate((x_f,i),axis=0)
        
    for i in y:
        y_f=np.concatenate((y_f,i),axis=0)
        
    x_5=np.vstack((x_f,y_f))
    
    return [[X_1, X_2, X_3, X_4, X_5], [x_1, x_2, x_3, x_4, x_5]]


    

#load images
def load_image(s,pts=[]):
    image=glob.glob(s+'.*')
    img=plt.imread(image[0])
    # fig,ax=plt.subplots()
    # ax.imshow(img, cmap='gray')
    # plt.show()
    return img


def load_images():
    images=glob.glob("images\*.tif")
    i=1
    for filename in images:
        img=plt.imread(filename)
        #corner=np.array([[[0,0]],[[1,1]]])
        #img = cv2.drawChessboardCorners(img, (1,1),corner,True)
        fig,ax=plt.subplots()
        ax.imshow(img,cmap='gray')
        x=eval('x_'+str(i)+'[0,:]')
        y=eval('x_'+str(i)+'[1,:]')
        i+=1
        ax.plot(x,y,'r.')
    
    



