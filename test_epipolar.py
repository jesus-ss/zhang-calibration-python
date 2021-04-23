
import numpy as np
import cv2 
from epipolar import *
from reproyect import image_plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_opencv_epipolar import main_load
from camera_calibration import calibrate_camera
from reproyect import reproyection_main
from load_epipolar import load_points

x_1,x_2,images1,images2,n_images=load_points()
n_points=int(len(x_1[0,:])/len(images1))
# x_1,x_2,images1,images2,n_points,Fopencv=main_load()
# n_points=len(x_1[0,:])

n_image=3

image_plot(images1[n_image],x_1[:,n_points*n_image:n_points*(n_image+1)])
image_plot(images2[n_image],x_2[:,n_points*n_image:n_points*(n_image+1)])

img1=images2[0]
fig,ax=plt.subplots()
ax.plot(x_1[0,n_points*n_image:n_points*(n_image+1)],x_1[1,n_points*n_image:n_points*(n_image+1)],'r.')
ax.plot(x_2[0,n_points*n_image:n_points*(n_image+1)],x_2[1,n_points*n_image:n_points*(n_image+1)],'b.')
ax.set_ylim(len(img1[0,:]),0)
ax.axis('equal')
plt.show(block=False)


F,P1,P2,X,x_1,x_2,e,M=compute_F(x_1,x_2)

rectify(images1[n_image],images2[n_image],F,e,M,x_1,x_2)
##### Compruebo las restricciones de F

#     -Simetria de P2 F P1
Fsimetrica=np.transpose(P2).dot(F.dot(P1))

inhomog=len(x_1)<3
if inhomog:
    x_1=np.concatenate((x_1,np.ones((1,len(x_1[0,:])))),axis=0)
inhomog=len(x_2)<3
if inhomog:
    x_2=np.concatenate((x_2,np.ones((1,len(x_2[0,:])))),axis=0)
 
#     -x2 F x=0    
xresto=[]
for i in range(n_points):
    xiresto=x_2[:,i].dot(F.dot(x_1[:,i]))
    xresto.append(xiresto)
xresto2=[]

# for i in range(n_points):
#     xiresto=x_2[:,i].dot(Fopencv.dot(x_1[:,i]))
#     xresto2.append(xiresto)



U,D,vt=np.linalg.svd(np.transpose(F))
e=vt[-1,:]
e=e/e[2]
plot_epipoles(F,x_1[:,n_points*n_image:n_points*(n_image+1)],x_2[:,n_points*n_image:n_points*(n_image+1)],epipole=e)
plot_epipoles(F,x_1[:,n_points*n_image:n_points*(n_image+1)],x_2[:,n_points*n_image:n_points*(n_image+1)],img=images2[n_image])
# plot_epipoles(F,x_1,x_2,epipole=e)
# plot_epipoles(F,x_1,x_2,img=images2[n_image])


min_axis=min([min(X[0,n_points*n_image:n_points*(n_image+1)]),min(X[1,n_points*n_image:n_points*(n_image+1)])])
max_axis=max([max(X[0,n_points*n_image:n_points*(n_image+1)]),max(X[1,n_points*n_image:n_points*(n_image+1)])])


fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[0,n_points*n_image:n_points*(n_image+1)],X[1,n_points*n_image:n_points*(n_image+1)],X[2,n_points*n_image:n_points*(n_image+1)])
ax.scatter(X[0,n_points*n_image:n_points*(n_image+1)],X[1,n_points*n_image:n_points*(n_image+1)],X[2,n_points*n_image:n_points*(n_image+1)])
ax.set_xlim3d(min_axis, max_axis)
ax.set_ylim3d(min_axis, max_axis)
#ax.set_zlim3d(min_axis, max_axis)
plt.show()



