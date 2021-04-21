import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
def optimal_solution(F,x_1,x_2):
    x1=x_1[0]
    y1=x_1[1]
    T1=np.array([[1,0,-x1],[0,1,-y1],[0,0,1]])
    x2=x_2[0]
    y2=x_2[1]
    T2=np.array([[1,0,-x2],[0,1,-y2],[0,0,1]])
    
    F_norm=np.linalg.inv(np.transpose(T2)).dot(F.dot(np.linalg.inv(T1)))
    u1,d1,vt1=np.linalg.svd(F_norm)
    e1=vt1[-1,:]
    u2,d2,vt2=np.linalg.svd(np.transpose(F_norm))
    e2=vt2[-1,:]
    e1=e1/(e1[0]**2+e1[1]**2)
    e2=e2/(e2[0]**2+e2[1]**2)
    
    R1=np.array([[e1[0],e1[1],0],[-e1[1],e1[0],0],[0,0,1]])
    R2=np.array([[e2[0],e2[1],0],[-e2[1],e2[0],0],[0,0,1]])
    # print(R1.dot(e1))
    # print(R2.dot(e2))
    
    F2=R2.dot(F_norm.dot(np.transpose(R1)))
    f1=e1[2]
    f2=e2[2]
    a=F2[1,1]
    b=F2[1,2]
    c=F2[2,1]
    d=F2[2,2]
    g=f2*c
    h=f2*d
    mp=-(a*d-b*c)
    p0=mp*b*d
    p1=mp*(a*d+b*c)+(b**4+h**4+2*b**2*h**2)
    p2=mp*(a*c+2*f1**2*b*d)+(4*a*b**3+4*g*h**3+4*a*b*h**2+4*b**2*g*h)
    p3=mp*(2*f1**2*a*d+2*f1**2*b*c)+(6*a**2*b**2+2*b**2*g**2+2*a**2*h**2+6*g**2*h**2+8*a*b*g*h)
    p4=mp*(2*a*c*f1**2+b*c*f1**4)+(4*a**3*b+4*a*b*g**2+4*a**2*g*h+4*g**3*h)
    p5=mp*(a*d*f1**4+b*c*f1**4)+(a**4+g**4)
    p6=a*c*f1**4
    
    # x=[i for i in range(-100,100)]
    # y=[]
    # for i in x:
    #     y.append(cost_fun(i,a,b,c,d,f1,f2))
    # fig=plt.figure()
    # plt.title('COST FUNCTION')
    # plt.scatter(x,y)
    # plt.show()
    
    p=[p6,p5,p4,p3,p2,p1,p0]
    roots=np.roots(p)
    asint=1/f1**2+c**2/(a**2+f2**2*c**2)
    
    min_val=asint
    cost_val=min_val+1
    t_min=0
    for i in roots:
        if np.isreal(i):
            cost_val=cost_fun(i,a,b,c,d,f1,f2)
        if cost_val<min_val:
            t_min=np.real(i)
    l1=[t_min*f1,1,-t_min]
    l2=[-f2*(c*t_min+d),a*t_min+b,c*t_min+d]
    x_1_min=np.array([-l1[0]*l1[2],-l1[1]*l1[2],l1[0]**2+l1[1]**2],dtype=float).reshape(3,1)
    x_2_min=np.array([-l2[0]*l2[2],-l2[1]*l2[2],l2[0]**2+l2[1]**2],dtype=float).reshape(3,1)
    x1_final=np.linalg.linalg.inv(T1).dot(np.transpose(R1).dot(x_1_min))
    x2_final=np.linalg.linalg.inv(T2).dot(np.transpose(R2).dot(x_2_min))
    return x1_final,x2_final

def cost_fun(t,a,b,c,d,f1,f2):
    s=t**2/(1+f1**2*t**2) + (c*t+d)**2/((a*t+b)**2+f2**2*(c*t+d)**2)
    return s
    



