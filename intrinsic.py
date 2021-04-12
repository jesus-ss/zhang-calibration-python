
import numpy as np
import scipy.optimize as optimize



def comp_homography2(m,M):
    
    m_norma,T2=normalise(m)
    M_norma,T1=normalise(M)
    
    npoints=len(m[0,:])
    A=np.zeros((npoints*3,9))
    O=[0,0,0]

    for n in range(0,npoints):
        X=M_norma[:,n]
        x,y,w=m_norma[0,n],m_norma[1,n],m_norma[2,n]
        A[3*n,0:3]=O
        A[3*n,3:6]=-w*X
        A[3*n,6:9]=y*X
        
        A[3*n+1,0:3] = w*X
        A[3*n+1,3:6] = O
        A[3*n+1,6:9] = -x*X
        
        A[3*n+2 ,0:3] = -y*X
        A[3*n+2 ,3:6] = x*X
        A[3*n+2 ,6:9] = O

    [U,D,V]=np.linalg.svd(A)
    V=np.transpose(V)
    H1 = V[:,8].reshape(3,3)   
    H2= np.linalg.inv(T2).dot(H1).dot(T1);
    H=H2/H2[2,2];
    
    #Utilizo scipy.optimize.leastsq para minimizar la distancia algebraica definida 
    #por la funcion projection_error(H,m,M)
    x0=H.reshape(1,9).ravel()   
    x=optimize.least_squares(projection_error,x0,args=(m,M),method='lm'
                              ,verbose=0)
    
    xvect=x['x']
    x=np.zeros((1,9))
    x[0,:]=xvect
    x=x.reshape(3,3)
    x=x/x[2,2]

    return x,H

def compute_A(H_mat):
    V=np.zeros((2*len(H_mat),6))
    
    j=0
    for hprima in H_mat:
        h=np.transpose(hprima)
        v12=np.array([h[0,0]*h[1,0],
             h[0,0]*h[1,1]+h[0,1]*h[1,0],
             h[0,1]*h[1,1],
             h[0,2]*h[1,0]+h[0,0]*h[1,2],
             h[0,2]*h[1,1]+h[0,1]*h[1,2],
             h[0,2]*h[1,2]])
        v11=np.array([h[0,0]*h[0,0],
             h[0,0]*h[0,1]+h[0,1]*h[0,0],
             h[0,1]*h[0,1],
             h[0,2]*h[0,0]+h[0,0]*h[0,2],
             h[0,2]*h[0,1]+h[0,1]*h[0,2],
             h[0,2]*h[0,2]])
        v22=np.array([h[1,0]*h[1,0],
             h[1,0]*h[1,1]+h[1,1]*h[1,0],
             h[1,1]*h[1,1],
             h[1,2]*h[1,0]+h[1,0]*h[1,2],
             h[1,2]*h[1,1]+h[1,1]*h[1,2],
             h[1,2]*h[1,2]])
        V[2*j,:]=v12
        V[2*j+1,:]=v11-v22
        j+=1
    K=np.transpose(V).dot(V)
    U,D,Vt=np.linalg.svd(K)
    Vt=np.transpose(Vt)
    b=Vt[:,5]
    # el vector b esta relacionado conla matriz B como:
    #    b=[B11,B12,B22,B13,B23,B33]

    B11=b[0]
    B12=b[1]
    B22=b[2]    
    B13=b[3] 
    B23=b[4] 
    B33=b[5] 
    
    #Los pramatros intrinsecos se caluclan sabiendo que B=A^(-T)*A^-1
    #APENCIDE B articulo a flexible new technique for camera calibration
    
    param=np.zeros(6)
    param[0]=(B12*B13 - B11*B23)/(B11*B22 - B12**2) 
    param[1]=B33 - (B13**2+ param[0]*(B12*B13 - B11*B23))/B11
    param[2]=np.sqrt(param[1]/B11)
    param[3]=np.sqrt(param[1]*B11/(B11*B22-B12**2))
    param[4]=-B12*param[2]**2*param[3]/param[1]
    param[5]=param[4]*param[0]/param[3] - B13*param[2]**2/param[1]
    
    # v0=(B12*B13 − B11*B23)/(B11*B22−B22**2) 
    # lmda = B33 − (B13**2+ v0*(B12*B13 − B11*B23))/B11
    # alfa=np.sqrt(lmda/B11)
    # beta=np.sqrt(lmda*B11/(B11*B22-B12**2))
    # gamma=-B12*alfa**2*beta/lmda
    # u0=gamma*v0/beta - B13*alfa**2/lmda
    #A= [alfa  gamma u0]
    #   [0     beta  v0]
    #   [0     0     1 ]

    A=np.array([[param[2],param[4],param[5],
                 0,param[3],param[0],
                 0,0,1]]).reshape(3,3)
    return A,param
    

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


def projection_error(H,m,M):
    h=np.zeros((1,9))
    h[0,:]=H
    h=h.reshape(3,3)
    
    inhomog=len(m)<3
    if inhomog:
        m=np.concatenate((m,np.ones((1,len(m[0,:])))),axis=0)
        
    inhomog=len(M)<3
    if inhomog:
        M=np.concatenate((M,np.ones((1,len(M[0,:])))),axis=0)
    
    X=np.zeros((len(M[:,0]),len(M[0,:])))
    j=0
    for i in np.transpose(M):
        X[:,j]=h.dot(i)
        j+=1
    
    X[0,:]=X[0,:]/X[2,:]
    X[1,:]=X[1,:]/X[2,:]
    X[2,:]=X[2,:]/X[2,:]
    
    #puntos reales (m) menos puntos proyectados a partir de la homografia (X) 
    res=m-X;
    req=np.concatenate((res[0,:],res[1,:]),axis=0) 
    return req



















