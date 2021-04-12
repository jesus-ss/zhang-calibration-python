
import numpy as np
import scipy.optimize as optimize

def compute_extrinsic(A,H,distor=True):
    lamd=1/np.linalg.norm(np.linalg.inv(A).dot(H[:,0]))  
    r1=lamd*np.linalg.inv(A).dot(H[:,0])
    r2=lamd*np.linalg.inv(A).dot(H[:,1])
    r3=np.cross(r1,r2)
    t=lamd*np.linalg.inv(A).dot(H[:,2])
    R=np.concatenate((r1.reshape(3,1),r2.reshape(3,1),r3.reshape(3,1),),axis=1)

    
    #Aproximo R a una matriz de rotacion 
    #APENDICE C
    U,D,V=np.linalg.svd(R)
    R=U.dot(V)
    
        
    return R,t

def compute_distortion(R_mat,t_mat,A,m_mat,M_mat):   
    D=[]
    d=[]
    param=[]
    for j in range(0,len(R_mat)):
        t=t_mat[j]
        R=R_mat[j]
        
        
        #construyo los parametros de R para la optimizacion
        q=deconsR(R)
        for k in q:
            param.append(k)
        #construyo los parametros de t para la optimizacion
        for k in t:
            param.append(k)
    
    
    #Termino de construir param con A y k1,k2
    for i in deconsA(A):
        param.append(i)
    for i in range(2):
        param.append(0)
    

    #posiciones de los parametros de A y k1,k2
    A_i=len(param)-7
    k_i=len(param)-2
    
    n_images=len(R_mat)
    x=optimize.least_squares(projection_error2,param,args=(m_mat,M_mat,n_images,A_i,k_i)
                              ,method='lm',verbose=0,ftol=1e-6,xtol=1e-6)
    

    #Reconstruyo de nuevo las matrices a partir de los parametros:
    param_opt=x['x']
    A_final=reconsA([param_opt[i] for i in range(A_i,A_i+5)])
    k1,k2=param_opt[k_i],param_opt[k_i+1]
    R_mat_final=[]
    t_mat_final=[]
    for i in range(0,n_images):
        Ri=reconsR([param_opt[k] for k in range(6*i,6*i+3)])
        R_mat_final.append(Ri)
        
        ti=np.array([param_opt[k] for k in range(6*i+3,6*i+6)])
        t_mat_final.append(ti)
    
    return A_final,k1,k2,R_mat_final,t_mat_final
    

def projection_error2(param,m_mat,M_mat,n_images,A_i,k_i):  
    A=reconsA([param[i] for i in range(A_i,A_i+5)])
    k=[param[i] for i in range(k_i,k_i+2)]
    R_mat=[]
    t_mat=[]
    res=np.array([0,0]).reshape(2,1)
    for i in range(0,n_images):
        R=reconsR([param[k] for k in range(6*i,6*i+3)])
        R_mat.append(R)
        
        t=np.array([param[k] for k in range(6*i+3,6*i+6)])
        t_mat.append(t)
        
    
    for j in range(0,len(m_mat)):
        t=t_mat[j]
        R=R_mat[j]
        m=m_mat[j]
        M=M_mat[j]
        #calculo u_prima a partir de k1,k2, A y Rt
        #el residuo es la deiferencia:
        #  res=[u_prima,v_prima]-m
        
        #construyo Rt=[r1 r2 t]
        Rt=np.concatenate((R[:,0].reshape(3,1),R[:,1].reshape(3,1),
                           t.reshape(3,1)),axis=1)
        #Puntos de la imagen XY
        # [x,y,z,w]=[R t]*[X,Y,Z,W]
        XY=np.zeros((len(M[:,0]),len(M[0,:])))
    
        l=0
        for i in np.transpose(M):
            XY[:,l]=Rt.dot(i)
            l+=1
        
        XY[0,:]=XY[0,:]/XY[2,:]
        XY[1,:]=XY[1,:]/XY[2,:]
        XY[2,:]=XY[2,:]/XY[2,:]
        
        #Puntos de la proyeccion de la imagen
        # [u,v,1]=A*[x,y,z]
        
        UV=np.zeros((len(M[:,0]),len(M[0,:])))
        l=0
        for i in np.transpose(XY):
            UV[:,l]=A.dot(i)
            l+=1
        
        UV[0,:]=UV[0,:]/UV[2,:]
        UV[1,:]=UV[1,:]/UV[2,:]
        UV[2,:]=UV[2,:]/UV[2,:]
        
        #La ecuacion para calcular los pixeles tras la distorsion es:
        #  u_prima=u+(u-u0)[k1(x^2+y^2)+k2(x^2+y^2)^2]
        #  v_prima=u+(v-v0)[k1(x^2+y^2)+k2(x^2+y^2)^2]
        
        u_prima=np.zeros(len(M[0,:]))
        v_prima=np.zeros(len(M[0,:]))
        u0=A[0,2]
        v0=A[1,2]
        k1=k[0]
        k2=k[1]
        for i in range(0,len(M[0,:])):
            u_prima[i]=UV[0,i]+(UV[0,i]-u0)*( k1*(XY[0,i]**2+XY[1,i]**2)+
                                             k2*(XY[0,i]**2+XY[1,i]**2)**2)
            v_prima[i]=UV[1,i]+(UV[1,i]-v0)*( k1*(XY[0,i]**2+XY[1,i]**2)+
                                             k2*(XY[0,i]**2+XY[1,i]**2)**2)
        uv_prima=np.vstack((u_prima,v_prima))
        
        res_i=m-uv_prima
        res=np.concatenate((res,res_i),axis=1)  
    req=np.concatenate((res[0,:],res[1,:]),axis=0) 
    return req
  
def deconsR(R):
    q1=-np.arcsin(R[0,2])
    q2=np.arcsin(R[0,1]/np.cos(q1))
    q3=np.arcsin(R[1,2]/np.cos(q1))
    
    return [q1,q2,q3]
    
def reconsR(angles):
    q1,q2,q3=angles[0],angles[1],angles[2]
    r=np.array([[np.cos(q1)*np.cos(q2), np.sin(q2)*np.cos(q1), -np.sin(q1)],
       [-np.sin(q2)*np.cos(q3)+np.cos(q2)*np.sin(q1)*np.sin(q3),
        np.cos(q2)*np.cos(q3)+np.sin(q2)*np.sin(q1)*np.sin(q3), np.cos(q1)*np.sin(q3)],
       [np.sin(q2)*np.sin(q3)+np.cos(q2)*np.sin(q1)*np.cos(q3), 
        -np.cos(q2)*np.sin(q3)+np.sin(q2)*np.sin(q1)*np.cos(q3), np.cos(q1)*np.cos(q3)]])
    R=r.reshape(3,3)
    return R

def deconsA(A):
    alfa=A[0,0]
    gamma=A[0,1]
    beta=A[1,1]
    u0=A[0,2]
    v0=A[1,2]
    return [alfa,gamma,beta,u0,v0]

def reconsA(para):
    alfa,gamma,beta,u0,v0=para[0],para[1],para[2],para[3],para[4]
    A=np.zeros((3,3))
    A[0,0]=alfa
    A[0,1]=gamma
    A[1,1]=beta
    A[0,2]=u0
    A[1,2]=v0 
    A[2,2]=1
    return A
    
### LEGACY CODE: NO ES NECESARIO, ES MAS, ES INCONVENIENTE, INICIALIZAR k1,k2,
### MEDIANTE ESTE METODO    
        # m=m_mat[j]
        # M=M_mat[j]
        
        
        # #Puntos de la imagen XY
        # # [x,y,z]=[R t]*[X,Y,Z]
        # XY=np.zeros((len(M[:,0]),len(M[0,:])))
        # Rt=np.concatenate((R[:,0].reshape(3,1),R[:,1].reshape(3,1),
        #                    t.reshape(3,1)),axis=1)
        # k=0
        # for i in np.transpose(M):
        #     XY[:,k]=Rt.dot(i)
        #     k+=1

        # XY[0,:]=XY[0,:]/XY[2,:]
        # XY[1,:]=XY[1,:]/XY[2,:]
        # XY[2,:]=XY[2,:]/XY[2,:]
        
        # #Puntos de la proyeccion de la imagen
        # # [u,v,1]=A*[x,y,z]
        
        # UV=np.zeros((len(M[:,0]),len(M[0,:])))
        # k=0
        # for i in np.transpose(XY):
        #     UV[:,k]=A.dot(i)
        #     k+=1
        
        # UV[0,:]=UV[0,:]/UV[2,:]
        # UV[1,:]=UV[1,:]/UV[2,:]
        # UV[2,:]=UV[2,:]/UV[2,:]
        
        
        # #La ecuacion a resolver para los parametros de distorsion es:
        # #  [ (u-u0)(x^2+y^2)  (u-u0)(x^2+y^2)^2] [k1]= [u_prima -u]
        # #  [ (v-v0)(x^2+y^2)  (v-v0)(x^2+y^2)^2] [k2]= [v_prima -v]
        # #   D * k= d
        # #Construyo D a partir de cada uno de los puntos de M
        # u0=A[0,2]
        # v0=A[1,2]
        

        # for i in range(0,len(M[0,:])):
        #     Di=[0,0]
        #     Di[0]=(UV[0,i]-u0)*(XY[0,i]**2+XY[1,i]**2)
        #     Di[1]=(UV[0,i]-u0)*(XY[0,i]**2+XY[1,i]**2)**2
            
            
        #     D.append(Di)
            
        #     Di=[0,0]
        #     Di[0]=(UV[1,i]-v0)*(XY[0,i]**2+XY[1,i]**2)
        #     Di[1]=(UV[1,i]-v0)*(XY[0,i]**2+XY[1,i]**2)**2
            
        #     D.append(Di)
        
        # #Construyo d:
        # di=0
        # for i in range(0,len(UV[0,:])):
        #     di=m[0,i]-UV[0,i]
        #     d.append(di)
        #     di=m[1,i]-UV[1,i]
        #     d.append(di)
# D=np.array(D)
#     d=np.array(d)
    
#     #  D * k= d 
#     #  k=(D^T * D)^-1 * D^T * d
#     k=np.linalg.inv(np.transpose(D).dot(D)).dot(np.transpose(D).dot(d))