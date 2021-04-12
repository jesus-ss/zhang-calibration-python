
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from intrinsic import normalise


def reproyection_main(img,A_final,k1,k2,Rotation,trasl):
    t=-0.94
    xcenter=0.1
    ycenter=-0.2
    tadjusted=trasl+np.array([(trasl[0]-xcenter)*t,(trasl[1]-ycenter)*t,trasl[2]*t])    
    Rt=np.concatenate((Rotation[:,0].reshape(3,1),Rotation[:,1].reshape(3,1),tadjusted.reshape(3,1)),axis=1)
    #reproyect(img,A_final,k1,k2)
    I_rect=reproyect(img,A_final,k1,k2,Rt)
    return I_rect
    
def image_plot(img,m=[]):
    if len(m):
        fig,ax=plt.subplots()
        ax.plot(m[0,:],m[1,:],'b.')
        ax.imshow(img,cmap='gray')
        plt.show()
    else:
        fig,ax=plt.subplots()
        ax.imshow(img,cmap='gray')
        plt.show()
         
          

def reproyect(I,A,k1,k2,Rt=np.identity(3)):
    [ix,iy]=np.meshgrid(range(len(I[0,:])),range(len(I[:,0])))    
    nr=len(I[:,0])
    nc=len(I[0,:])
    I_rect=np.ones((nr,nc))
    
    px=ix.reshape(len(ix[:,0])*len(ix[0,:]))
    py=iy.reshape(len(iy[:,0])*len(iy[0,:]))
    pxy=np.vstack((px,py,np.ones(len(ix[:,0])*len(ix[0,:]))))
    pxy=np.array(pxy,dtype='int')


    xy=np.zeros((len(pxy[:,0]),len(pxy[0,:])))
    l=0
    for i in np.transpose(pxy):
        xy[:,l]=(np.linalg.inv(A)).dot(i)
        l+=1

    xy2=np.zeros((len(pxy[:,0]),len(pxy[0,:])))
    l=0
    for i in np.transpose(xy):
        xy2[:,l]=Rt.dot(i)
        l+=1
        
    
    uv=np.zeros((len(pxy[:,0]),len(pxy[0,:])))
    l=0
    for i in np.transpose(xy2):
            uv[:,l]=A.dot(i)
            l+=1
    xy2[0,:]=xy2[0,:]/xy2[2,:]
    xy2[1,:]=xy2[1,:]/xy2[2,:]
    xy2[2,:]=xy2[2,:]/xy2[2,:]
    
    uv[0,:]=uv[0,:]/uv[2,:]
    uv[1,:]=uv[1,:]/uv[2,:]
    uv[2,:]=uv[2,:]/uv[2,:]
    
    pxy2=distort(xy2,uv,k1,k2,A)

    pxy2[0,:]=pxy2[0,:]/pxy2[2,:]
    pxy2[1,:]=pxy2[1,:]/pxy2[2,:]
    pxy2[2,:]=pxy2[2,:]/pxy2[2,:]

    
    print(max(xy2[0,:]),min(xy2[0,:]))
    print(max(xy2[1,:]),min(xy2[1,:]))
    print(max(xy2[2,:]),min(xy2[2,:]))

    
    good_points=[i for i in range(len(pxy2[0,:])) if 
                 (pxy2[0,i]<(nc-2) and pxy2[0,i]>0 
                  and pxy2[1,i]>0 and pxy2[1,i]<(nr-2))]
    print(len(good_points))
    
    pxy2=np.transpose(np.array([pxy2[:,i] for i in good_points]))
    pxy=np.transpose(np.array([pxy[:,i] for i in good_points]))

    #Como el resultado de las transformaciones no sera un numero entero, tengo que 
    #interpolar las coordenadas del pixel concreto a partir de sus primeros vecinos
    #lu=arriba-izquierda ru=arriba-derecha
    #ld=abajo-izquierda  rd=abajo-derecha
    #alfa=proprcion de pixel mas proximo en cada direccion
    pxy0=np.floor(pxy2) 
    pxy0=np.array(pxy0,dtype='int')
    alfa=pxy2-pxy0
    alfax=alfa[0,:]
    alfay=alfa[1,:]
    
    px0=pxy0[0,:]
    py0=pxy0[1,:]
    
    ld=np.vstack((px0,py0+1))
    rd=np.vstack((px0+1,py0+1))
    ru=np.vstack((px0+1,py0))
    lu=np.vstack((px0,py0))
    
    for n in range(len(pxy[0,:])):
        u=pxy[0,n]
        v=pxy[1,n]

        I_rect[v,u]=(1-alfax[n])*(1-alfay[n])*I[lu[1,n],lu[0,n]]+(1-alfax[n])*alfay[n]*I[ld[1,n],ld[0,n]]+(1-alfay[n])*alfax[n]*I[ru[1,n],ru[0,n]]+alfax[n]*alfay[n]*I[rd[1,n],rd[0,n]]
     
    return I_rect
    
def distort(XY,UV,k1,k2,A):
    u_prima=np.zeros(len(XY[0,:]))
    v_prima=np.zeros(len(XY[0,:]))
    u0=A[0,2]
    v0=A[1,2]

    for i in range(len(XY[0,:])):
        u_prima[i]=UV[0,i]+(UV[0,i]-u0)*( k1*(XY[0,i]**2+XY[1,i]**2)+
                                         k2*(XY[0,i]**2+XY[1,i]**2)**2)
        v_prima[i]=UV[1,i]+(UV[1,i]-v0)*( k1*(XY[0,i]**2+XY[1,i]**2)+
                                         k2*(XY[0,i]**2+XY[1,i]**2)**2)
    uv_prima=np.vstack((u_prima,v_prima,UV[2,:]))
    
    
    
    
    
    # x_prima=np.zeros(len(XY[0,:]))
    # y_prima=np.zeros(len(XY[0,:]))
    
    
    
    # #  x_prima=x+x[k1(x^2+y^2)+k2(x^2+y^2)^2]
    # #  y_prima=y+y[k1(x^2+y^2)+k2(x^2+y^2)^2]
    # for i in range(0,len(XY[0,:])):
    #     x_prima[i]=XY[0,i]+XY[0,i]*( k1*(XY[0,i]**2+XY[1,i]**2)+
    #                                      k2*(XY[0,i]**2+XY[1,i]**2)**2)
    #     y_prima[i]=XY[1,i]+XY[1,i]*( k1*(XY[0,i]**2+XY[1,i]**2)+
    #                                      k2*(XY[0,i]**2+XY[1,i]**2)**2)
    # xy_prima=np.vstack((x_prima,y_prima,XY[2,:]))

    return uv_prima
    
    