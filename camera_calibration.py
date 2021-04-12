
import numpy as np
from intrinsic import *
from extrinsic import *

def calibrate_camera(x,X):
    if len(x)<5:
        return error
    
    H_mat=[]
    H_guess_mat=[]
    for i in range(0,len(x)):    
        H,H_guess=comp_homography2(x[i],X[i])
        H_guess_mat.append(H_guess)
        H_mat.append(H)
    
    A,param=compute_A(H_mat)
    
    R_mat=[]
    t_mat=[]
    
    for i in range(0,len(x)):
        R,t=compute_extrinsic(A,H_mat[i])
        R_mat.append(R)
        t_mat.append(t)
        
    A_final,k1,k2,R_mat_final,t_mat_final=compute_distortion(R_mat,t_mat,A,x,X)
    
    print("La distancia focal es:  [",A_final[0,0], ",", A_final[1,1],"]")
    print("El punto principal es:  [" ,A_final[0,2], ",", A_final[1,2],"]")
    print("Skew:  ",A_final[0,1])
    
    return A_final,k1,k2,R_mat_final,t_mat_final