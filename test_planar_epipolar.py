
import numpy as np
from intrinsic import comp_homography2
from load_epipolar import load_points



x_1,x_2,images1,images2,n_images=load_points()
n_points=len(x_1[0,:])

### Homografia x2 ---> x1
H,H1=comp_homography2(x_2,x_1)

inhomog=len(x_1)<3
if inhomog:
    x_1homo=np.concatenate((x_1,np.ones((1,len(x_1[0,:])))),axis=0)
inhomog=len(x_2)<3
if inhomog:
    x_2homo=np.concatenate((x_2,np.ones((1,len(x_2[0,:])))),axis=0)
####
    
x_22=[]
for i in range(n_points):
    xi_22=H.dot(x_1homo[:,i])
    x_22.append(xi_22)
x_22=np.array(x_22)
x_22=np.transpose(x_22)
x_22[0,:]=x_22[0,:]/x_22[2,:]
x_22[1,:]=x_22[1,:]/x_22[2,:]
x_22[2,:]=x_22[2,:]/x_22[2,:]


### Puntos de corte entre x_22 y x_2 es el epipolo
