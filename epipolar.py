
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from triangulation import *
from intrinsic import comp_homography2
from reproyect import reproyect2
from copy import copy

def compute_F(x_1,x_2):
    inhomog=len(x_1)<3
    if inhomog:
        x_1=np.concatenate((x_1,np.ones((1,len(x_1[0,:])))),axis=0)
    inhomog=len(x_2)<3
    if inhomog:
        x_2=np.concatenate((x_2,np.ones((1,len(x_2[0,:])))),axis=0)
    #Construyo la matriz de ecuaciones M tal que M*f=0
    #x2x1f11+x2y1f12+x2f13+y2x1f21+y2y1f22+y2f23+xf31+yf32+f33=0
    
    x1_norm,T1=normalise(x_1)
    x2_norm,T2=normalise(x_2)
    n_points=len(x_1[0,:])
    M=np.zeros((n_points,9))
    for i in range(n_points):
        x2,y2,w2=x2_norm[:,i]
        x1,y1,w1=x1_norm[:,i]
        # x2,y2,w2=x_2[:,i]
        # x1,y1,w1=x_1[:,i]
        
        M[i,:]=[x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1]
    # fig=plt.figure()
    # plt.scatter(x1_norm[0,:42],x1_norm[1,:42])
    # plt.scatter(x1_norm[0,42:84],x1_norm[1,42:84])
    # plt.scatter(x1_norm[0,84:84+42],x1_norm[1,84:84+42])
    # fig=plt.figure()
    # plt.scatter(x2_norm[0,:42],x2_norm[1,:42])
    # plt.scatter(x2_norm[0,42:84],x2_norm[1,42:84])
    # plt.scatter(x2_norm[0,84:84+42],x2_norm[1,84:84+42])
    U1,D1,VT1=np.linalg.svd(M)
    f=VT1[-1,:]
    F=f.reshape((3,3))
    #F=F/F[2,2]
    
    
    #aplico la restirccion del rango 2 de la matriz F
    U2,d2,VT2=np.linalg.svd(F)
    D2=np.diag(d2)
    D2[2,2]=0
    F1=U2.dot(D2.dot(VT2))
    
    #desnormalizo F
    F2=np.transpose(T2).dot(F1.dot(T1))
    # F2=F1
    #F2=F2/F2[2,2]
    #construyo las matrices de las camaras a partir de F y estimo los puntos 
    #del mundo X
    P1,P2=construct_P(F2)
    x1_f=[]
    x2_f=[]
    for i in range(n_points):
        x1i_f,x2i_f=optimal_solution(F2,x_1[:,i],x_2[:,i])
        x1i_f=x1i_f/x1i_f[2]
        x1_f.append(x1i_f)
        x2i_f=x2i_f/x2i_f[2]
        x2_f.append(x2i_f)
    
    x1_f=np.transpose(np.array(x1_f)) 
    x1_f=x1_f[0,:,:]
    x2_f=np.transpose(np.array(x2_f))    
    x2_f=x2_f[0,:,:]
    X=compute_X(P1,P2,x1_f,x2_f)
    X[0,:]=X[0,:]/X[3,:]  
    X[1,:]=X[1,:]/X[3,:] 
    X[2,:]=X[2,:]/X[3,:] 
    X[3,:]=X[3,:]/X[3,:]
    # #A partir de este punto empieza la parte de optimizacion del algoritmo
    param=[]
    for i in P2.ravel():
        param.append(i)
    X2=np.vstack((X[0,:],X[1,:],X[2,:]))
    for i in X2:
        for j in i:
            param.append(j)
    opt=optimize.least_squares(projection_error3,param,args=(x1_f,x2_f,P1),method='lm')
    x=opt['x']
    p2=x[0:12]
    Xi=np.array(x[12:12+n_points])
    Y=np.array(x[12+n_points:12+2*n_points])
    Z=np.array(x[12+2*n_points:12+3*n_points])
    W=np.ones((n_points))
    X_final=np.vstack((Xi,Y,Z,W))
    
    #Reconstruyo la matriz funamental final a partir de P2 optimizado
    P22=p2.reshape(3,4)
    
    M=P22[0:3,0:3]
    t=P22[:,3]
    
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    F_final=tx.dot(M)
    F_final=F_final/F_final[2,2]
    return F_final,P1,P2,X_final,x1_f,x2_f,t,M


    
def plot_epipoles(F,x_1,x_2,img=[],epipole=[0,0]):
    fig,ax=plt.subplots()
    x_min=min([epipole[0],min(x_2[0,:])])
    if len(img):
        x_max=max([epipole[0],max(x_2[0,:]),len(img[0,:])])
        ax.imshow(img,cmap='gray')
        ax.set_ylim(len(img[:,0]),0)
    else:
        x_max=max([epipole[0],max(x_2[0,:])])
        plt.gca().invert_yaxis()

    n_points=len(x_1[0,:])
    for i in range(len(x_1[0,:])):
        xi=np.ones((3,1))
        xi[0:2,0]=x_1[:2,i]
        u=F.dot(xi)
        x=np.array(([i for i in range(int(x_min),int(x_max))]),dtype=float)
        y=-(u[0]*x+u[2])/u[1]
        ax.plot(x,y,'b')   
    ax.plot(x_2[0,:],x_2[1,:],'k.')
    ax.plot(epipole[0],epipole[1],'r.')       
    plt.axis('scaled')
    plt.show(block=False)
    

def projection_error3(param,x1,x2,P1):
    n_points=len(x1[0,:])
    p2=np.array(param[0:12])
    P2=p2.reshape(3,4)
    x=np.array(param[12:12+n_points])
    y=np.array(param[12+n_points:12+2*n_points])
    z=np.array(param[12+2*n_points:])
    w=np.ones((n_points))
    X=np.vstack((x,y,z,w))

    x1_prima=np.zeros((3,n_points))
    x2_prima=np.zeros((3,n_points))
    for i in range(n_points):
        x1_prima[:,i]=P1.dot(X[:,i])
    for i in range(n_points):
        x2_prima[:,i]=P2.dot(X[:,i]) 
    x2_prima[0,:]=x2_prima[0,:]/x2_prima[2,:]
    x2_prima[1,:]=x2_prima[1,:]/x2_prima[2,:]
    x2_prima[2,:]=x2_prima[2,:]/x2_prima[2,:]
    x1_prima[0,:]=x1_prima[0,:]/x1_prima[2,:]
    x1_prima[1,:]=x1_prima[1,:]/x1_prima[2,:]
    x1_prima[2,:]=x1_prima[2,:]/x1_prima[2,:]
    res1=(x2_prima-x2)**2
    res2=(x1_prima-x1)**2
    resq=np.concatenate((res1[0,:],res1[1,:],res2[0,:],res2[1,:]))
    return resq
    
def compute_X(P1,P2,x_1,x_2):
    #Filas de P1:
    p1_1=P1[0,:]
    p1_2=P1[1,:]
    p1_3=P1[2,:]
    #Filas P2:
    p2_1=P2[0,:]
    p2_2=P2[1,:]
    p2_3=P2[2,:]
    n_points=len(x_1[0,:])
    X=[]
    for i in range(n_points):
        x1=x_1[0,i]
        y1=x_1[1,i]
        w1=x_1[2,i]
        x2=x_2[0,i]
        y2=x_2[1,i]
        w2=x_2[2,i]
        
        A=np.zeros((6,4))
        A[0,:]=x1*p1_3-p1_1
        A[1,:]=y1*p1_3-p1_2
        A[2,:]=x2*p2_3-p2_1
        A[3,:]=y2*p2_3-p2_2
        A[4,:]=x1*p1_2-y1*p1_1
        A[5,:]=x2*p2_2-y2*p2_1
        
        A2=np.transpose(A).dot(A)
        u,d,vt=np.linalg.svd(A)
        
        X.append(vt[-1,:])
    X=np.array(X)
    
    return np.transpose(X)
def construct_P(F):
    P1=np.concatenate((np.identity(3),np.zeros((3,1))),axis=1)
    
    U,D,vt=np.linalg.svd(np.transpose(F))
    e=vt[-1,:]
    e=e/e[2]
    ex=np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
    P2=np.concatenate((ex.dot(F),e.reshape(3,1)),axis=1)
    return P1,P2

def normalise(m):
    x=np.zeros((len(m[:,0]),len(m[0,:])))
    
    
    for i in range(0,len(m[:,0])):
        x[i,:]=m[i,:]
    
    
    
    inhomog=len(x)<3
    if inhomog:
        x=np.concatenate((x,np.ones((1,len(x[0,:])))),axis=0)
    x[0,:]=x[0,:]/x[2,:]
    x[1,:]=x[1,:]/x[2,:]
    
    #calcular centroide
    xm=np.mean(x[0,:])
    ym=np.mean(x[1,:])
    
    #muevo los puntos respecto al nuevo centro
    x_new=np.vstack((x[0,:]-xm,x[1,:]-ym,x[2,:]))

    
    meandist = np.mean(np.sqrt(np.square(x_new[0,:]) + np.square(x_new[1,:])))
    scale=np.sqrt(2)/meandist
    
    #matriz de normalizacion, explicada en el libro de Faugeras pag 42.
    #Tiene la forma:
    #T=[s  t]
    #  [0  1]
    T=np.zeros((3,3))
    T[0,:]=[scale,0,-scale*xm]
    T[1,:]=[0,scale,-scale*ym]
    T[2,:]=[0,0,1]
    
    #Multiplico los vectores de cada punto [x,y,w] por la matriz:
    j=0
    for i in np.transpose(x):
        x[:,j]=T.dot(i)       
        j+=1  
    return x,T

def rectify(img1,img2,F,e,M,x_1,x_2): 

    n_points=len(x_1[0,:])
    
    center=[len(img1[0,:])/4,len(img1[:,0])/4]
    T=np.identity(3)
    T[0,2]=center[0]
    T[1,2]=center[1]
    R=np.identity(3)
    alfa=np.arctan(e[1]/e[0])
    R[0,0]=np.cos(-alfa)
    R[1,1]=R[0,0]
    R[1,0]=np.sin(-alfa)
    R[0,1]=-R[1,0]
    
    e1=R.dot(e)
    f=e1[0]
    G=np.identity(3)
    G[2,0]=-1/f
    H_prima=G.dot(R)
    #H_prima=H_prima/H_prima[2,2]
    e0=H_prima.dot(e)
    
    param=[0,0,0,0,0,0,0,0]
    x=optimize.least_squares(projection_error5,param,args=(x_1,x_2,H_prima),method='lm')

    para=x['x']
    h=np.append(para,1)
    H2=h.reshape(3,3)
    
    I_rect2=reproyect2(img2,H_prima)  
    I_rect1=reproyect2(img1,H2)


    fig,ax=plt.subplots()
    
    # x_min=0
    # x_max=len(img1[0,:])
    # for i in range(len(x_1[0,:])):
    #     xi=np.ones((3,1))
    #     xi[0:2,0]=x_1[:2,i]
    #     u=F.dot(xi)
    #     x=np.array(([i for i in range(0,612)]),dtype=float)
    #     y=-(u[0]*x+u[2])/u[1]
    #     #ax.plot(x,y,'r')
    #     xy1=np.vstack((x,y))
    #     xy=apply_H(xy1,H_prima)
    #     ax.plot(xy[0,:],xy[1,:],'b')
    #     xy2=apply_H(xy,np.linalg.inv(H_prima))
    #     ax.plot(xy2[0,:],xy2[1,:],'r')
    # plt.axis('scaled') 
    # plt.gca().invert_yaxis()   
    
    plt.imshow(I_rect1,cmap='gray')
    plt.show()
    
    fig=plt.figure()
    plt.imshow(I_rect2,cmap='gray')
    plt.show()


def projection_error5(para,x_1,x_2,H_prima):
    h=copy(para)
    h=np.concatenate((h,[1]))
    H=np.array(h).reshape(3,3)
    
    x1_prima=np.zeros((3,len(x_1[0,:])))
    x2_prima=np.zeros((3,len(x_1[0,:])))

    for i in range(len(x_1[0,:])):
        x1_prima[:,i]=H.dot(x_1[:,i])
    x1_prima[0,:]=x1_prima[0,:]/x1_prima[2,:]
    x1_prima[1,:]=x1_prima[1,:]/x1_prima[2,:]
    x1_prima[2,:]=x1_prima[2,:]/x1_prima[2,:]    
    
    for i in range(len(x_1[0,:])):
        x2_prima[:,i]=H_prima.dot(x_2[:,i])
    x2_prima[0,:]=x2_prima[0,:]/x2_prima[2,:]
    x2_prima[1,:]=x2_prima[1,:]/x2_prima[2,:]
    x2_prima[2,:]=x2_prima[2,:]/x2_prima[2,:]
    
    res=(x1_prima-x2_prima)**2
    resq=np.concatenate((res[0,:],res[1,:]))
    return resq
 
# def projection_error4(para,x_1,x_2,H_prima,H0):
#     a=para[0]
#     b=para[1]
#     c=para[2]
    
#     Ha=np.identity(3)
#     Ha[0,0]=a
#     Ha[0,1]=b
#     Ha[0,2]=c
    
#     H=Ha.dot(H0)
    
#     x1_prima=np.zeros((3,len(x_1[0,:])))
#     x2_prima=np.zeros((3,len(x_1[0,:])))
    
#     for i in range(len(x_1[0,:])):
#         x1_prima[:,i]=H.dot(x_1[:,i])
#     x1_prima[0,:]=x1_prima[0,:]/x1_prima[2,:]
#     x1_prima[1,:]=x1_prima[1,:]/x1_prima[2,:]
#     x1_prima[2,:]=x1_prima[2,:]/x1_prima[2,:]    
    
#     for i in range(len(x_1[0,:])):
#         x2_prima[:,i]=H_prima.dot(x_2[:,i])
#     x2_prima[0,:]=x2_prima[0,:]/x2_prima[2,:]
#     x2_prima[1,:]=x2_prima[1,:]/x2_prima[2,:]
#     x2_prima[2,:]=x2_prima[2,:]/x2_prima[2,:]
    
#     res=(x1_prima-x2_prima)**2
#     resq=np.concatenate((res[0,:],res[1,:]))
#     return resq

  
# def projection_error4(para,x_1,x_2):
#     n_points=len(x_1[0,:])
#     a=para[0]
#     b=para[1]
#     c=para[2]
#     d=para[3]
#     res=0
#     for i in range(n_points):
#         res=res+(a*x_1[0,i]+b*x_1[1,i]+c-x_2[0,i])**2+(x_1[0,i]*d-x_2[1,i])**2
#     return res
    
    # HA=np.identity(3)
    # HA[0,0]=a
    # HA[0,1]=b
    # HA[0,2]=c
    # n_points=len(x_1[0,:])
    # x_prima=np.zeros((3,n_points))
    
    # for i in range(n_points):
    #     x_prima[:,i]=HA.dot(x_1[:,i])
    
    # x_prima[0,:]=x_prima[0,:]/x_prima[2,:]
    # x_prima[1,:]=x_prima[1,:]/x_prima[2,:]
    # x_prima[2,:]=x_prima[2,:]/x_prima[2,:]

    # res=(x_2-x_prima)**2
    # resq=np.concatenate((res[0,:],res[1,:],res[2,:]),axis=0)
    return res

def apply_H(x,H):
    n_points=len(x[0,:])
    inhomog=len(x)<3
    if inhomog:
        x=np.concatenate((x,np.ones((1,len(x[0,:])))),axis=0)
    x_22=[]
    for i in range(n_points):
        xi_22=H.dot(x[:,i])
        x_22.append(xi_22)
    x_22=np.array(x_22)
    x_22=np.transpose(x_22)
    x_22[0,:]=x_22[0,:]/x_22[2,:]
    x_22[1,:]=x_22[1,:]/x_22[2,:]
    x_22[2,:]=x_22[2,:]/x_22[2,:]
    return x_22

# def rectify(img,x_1,x_2,F):
#     ## Calculo la homografia x' ----> x
#     H,H1=comp_homography2(x_1,x_2)
#     x_11=apply_H(x_2,H)
#     ### Calculo las lineas epipolares l2=F.x1 y aplico la homografia a cada 
#     ### uno de los puntos de las lineas epipolares
#     fig,ax=plt.subplots()
#     x_min=0
#     x_max=len(img[0,:,0])
#     for i in range(len(x_1[0,:])):
#         xi=np.ones((3,1))
#         xi[0:2,0]=x_1[:2,i]
#         u=F.dot(xi)
#         x=np.array(([i for i in range(int(x_min),int(x_max))]),dtype=float)
#         y=-(u[0]*x+u[2])/u[1]
#         ax.plot(x,y,'g')
#         xy1=np.vstack((x,y))
#         xy=apply_H(xy1,H)
#         x=xy[0,:]
#         y=xy[1,:]
#         # good_points1=np.where(x>0)
#         # x=x[good_points1]
#         # y=y[good_points1]
#         # good_points2=np.where(x<x_max)
#         # x=x[good_points2]
#         # y=y[good_points2]
#         ax.plot(x,y,'b')
#         ax.set_xlim(0,x_max)
        
#     plt.gca().invert_yaxis()
#     plt.axis('scaled')
#     ax.plot(x_11[0,:],x_11[1,:],'r.')
#     ax.plot(x_1[0,:],x_1[1,:],'k.')
#     ax.plot(x_2[0,:],x_2[1,:],'g.')
#     plt.show()
    







# def plane_rectify(F,x_1,x_2):
#     x1=x_1[:,10:13]
#     x2=x_2[:,10:13]
    
#     inhomog=len(x1)<3
#     if inhomog:
#         x1=np.concatenate((x1,np.ones((1,len(x1[0,:])))),axis=0)
#     inhomog=len(x2)<3
#     if inhomog:
#         x2=np.concatenate((x2,np.ones((1,len(x2[0,:])))),axis=0)
#     n_points=len(x1[0,:])
    
#     x1_prima=[]
#     x2_prima=[]
#     for i in range(n_points):
#         xi1=x1[:,i]
#         xi2=x2[:,i]
#         xi1_prima,xi2_prima=optimal_solution(F,xi1,xi2)
#         x1_prima.append(xi1_prima)
#         x2_prima.append(xi2_prima)
#     x1_prima=np.transpose(np.array(x1_prima)) 
#     x1_prima=x1_prima[0,:,:]
#     x2_prima=np.transpose(np.array(x2_prima))    
#     x2_prima=x2_prima[0,:,:]
#     x2_prima=x2_prima/x2_prima[2,:]

#     U,D,vt=np.linalg.svd(np.transpose(F))
#     e=vt[-1,:]
#     #e=e/e[2]
    
#     ex=np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
#     A=ex.dot(F)
    
#     M=np.vstack((x1_prima[:,0],x1_prima[:,1],x1_prima[:,2]))
#     b=np.zeros((n_points))
    
#     for i in range(n_points):
#         xi1=x1_prima[:,i]
#         xi2=x2_prima[:,i]
#         b[i]=( np.cross( xi2, (A.dot(xi1)) ).dot( np.cross(xi2,e) ) )/( ( np.cross(xi2,e) ).dot( np.cross(xi2,e) ) )
#     b=b/b[2]
#     # Mv=b   b x Mv=0
#     #filas de M mi son x1_prima[:,i]
#     # M2 . v=b x M . v
#     m1=x1_prima[:,0]
#     m2=x1_prima[:,1]
#     m3=x1_prima[:,2]
#     b1=b[0]
#     b2=b[1]
#     b3=b[2]
    
#     M2=np.zeros((3,3))
#     M2[0,:]=b1*m3-b3*m1
#     M2[1,:]=b2*m3-b3*m2
#     M2[2,:]=b1*m2-b2*m3
    
#     u,d,vt=np.linalg.svd(A)
#     v=vt[-1,:].reshape(1,3)
#     #v=v/v[0,2]
#     b2=M.dot(v.reshape(3,1))
#     b2=b2/b2[2]
#     e=e.reshape(1,3)
#     res=np.transpose(e).dot(v)
#     H=A-res
#     x11=[]
#     for i in range(n_points):
#         x11.append(H.dot(x2[:,i]))
    
#     x11=np.array(x11)
#     x11=np.transpose(x11)
#     x11[0,:]=x11[0,:]/x11[2,:]
#     x11[1,:]=x11[1,:]/x11[2,:]
#     x11[2,:]=x11[2,:]/x11[2,:]
    

    

        