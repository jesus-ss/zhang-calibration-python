# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:34:24 2021

@author: jesus
"""

import numpy as np


def comp_homography(m,M):
    #normalize point coordinates   
    Np=len(m[0,:])    
    m=np.concatenate((m,np.ones((1,Np))),axis=0)
    M=np.concatenate((M,np.ones((1,Np))),axis=0)

    ax=m[0,:]
    ay=m[1,:]
    mxx=np.mean(ax)
    myy=np.mean(ay)
    ax=ax-mxx
    ay=ay-myy
    scxx=np.mean(abs(ax))
    scyy=np.mean(abs(ay))
    
    Hnorm=np.matrix([[1/scxx,0,-mxx/scxx],[0,1/scyy,-myy/scyy],[0,0,1]])
    inv_Hnorm=np.matrix([[scxx, 0 , mxx], [0,scyy,myy], [0,0,1]])
    
    mn=np.zeros((3,Np))
    j=0
    for i in np.transpose(m):
        mn[:,j]=Hnorm.dot(i)
        j+=1
        

    L=np.zeros((2*Np,9))
    L[[i for i in range(0,Np*2,2)],0:3]=np.transpose(M)
    L[[i for i in range(1,Np*2,2)],3:6]=np.transpose(M)
    L[[i for i in range(0,Np*2,2)],6:9]=-np.transpose((np.ones((3,1))*mn[0,:])*M)
    L[[i for i in range(1,Np*2,2)],6:9]=-np.transpose((np.ones((3,1))*mn[1,:])*M)
    
    Lprima=np.transpose(L).dot(L)
    
    
    U,D,V=np.linalg.svd(Lprima)
    
    hh=V[8,:]
    hh=hh/hh[8] #h defined up to a scaling factor
    
        
    Hrem=hh.reshape(3,3)
    H=inv_Hnorm*Hrem
    return H