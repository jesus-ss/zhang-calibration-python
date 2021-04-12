
import numpy as np
from load_data import *
from reproyect import *
from camera_calibration import *


[X,x]=load_model()
A_final,k1,k2,R_mat_final,t_mat_final=calibrate_camera(x,X)

img_reproyected_mat=[]
for i in range(len(x)):
    img=load_image('images\CalibIm'+str(i+1)) 
    Rprueba=R_mat_final[i]
    trasl=t_mat_final[i]
    #reproyect(img,A_final,k1,k2)
    img_reproyected=reproyection_main(img,A_final,k1,k2,Rprueba,trasl)
    img_reproyected_mat.append(img_reproyected)

for i in img_reproyected_mat:
    image_plot(i)

#image_plot(img,x[4])
