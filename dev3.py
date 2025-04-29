import numpy as np
import matplotlib.pyplot as plt
from math import gamma

def filtration():
    # Bu dastur vaqt buyicha ikkinchi tartibli kasr hosila va 
    # vaqt bouyicha birinchi tartibli kasr hosilalarni hisoblaydi.

    k0 = 1e-13
    myu0 = 1e-2     
    Pc = 150e5
    Pk = 200e5
    bet_y = 3e-10

    tMax = 3600

    lv = 1000
    lp = 500
    betta = 0.5
    alf = 1

    bet = 1 + betta                 

    tau = 100
    h = 0.05
    L = 40          
    T = int(tMax/tau) + 1          
    N = int(L/h) + 1   

    rc = 0.1
    Nc = int(rc/h) + 1

    # Boshlang'ich va chegaraviy shartlar
    P = np.zeros((T, N))
    U = np.zeros((T, N))

    # chegaraviy shart chap taraf 
    P[:, :Nc] = Pc
    # boshlangich shart1
    P[0, Nc:] = Pk
    # chegaraviy shart ong taraf
    P[:, -1] = Pk

    kap = k0 / (myu0 * bet_y)

    for i in range(3, N):
        P[1, i] = P[0, i]  # bu joyi xato tekshir "-"

    kv = lv / ((tau**bet) * gamma(3 - bet))
    kp = lp / ((tau**alf) * (h**2) * gamma(2 - alf))
    
    kv1 = lv / ((tau**betta) * gamma(2 - betta))   # tezlik
    kp2 = lp / ((tau**alf) * gamma(2 - alf))   # tezlik
        
    alpha = np.zeros(N+1)
    beta = np.zeros(N+1)

    for j in range(1, T-1):
        for i in range(3, N-1):  
            sv = 0 
            for k in range(1, j):
                sv += (P[k+1, i] - 2*P[k, i] + P[k-1, i]) * ((j - k + 1)**(2 - bet) - (j - k)**(2 - bet))
            
            sp1 = 0
            sp2 = 0
            sp3 = 0
            st = 0
            for k in range(j):
                sp1 += (P[k+1, i+1] - P[k, i+1]) * ((j - k + 1)**(1 - alf) - (j - k)**(1 - alf))
                sp2 += (P[k+1, i] - P[k, i]) * ((j - k + 1)**(1 - alf) - (j - k)**(1 - alf))
                sp3 += (P[k+1, i-1] - P[k, i-1]) * ((j - k + 1)**(1 - alf) - (j - k)**(1 - alf))
                st += (U[k+1, i] - U[k, i]) * ((j - k + 1)**(1 - betta) - (j - k)**(1 - betta))  # tezlik
        
        alpha[4] = 0
        beta[4] = Pc
        for i in range(4, N-1):
            r1 = (2*i*h - h)/2
            r0 = (2*i*h - 3*h)/2
            A = kap * (r0/h**2 + kp*r0) / ((i-1)*h)  
            B = kap * ((r1 + r0)/h**2 + kp*(r1 + r0)) / ((i-1)*h) + kv + 1/tau 
            C = kap * (r1/h**2 + kp*r1) / ((i-1)*h) 
            F = P[j, i]/tau - kv*sv + 2*kv*P[j, i] - kv*P[j-1, i] + (kap*kp*r1*sp1)/((i-1)*h) - (kap*kp*r1*P[j, i+1])/((i-1)*h) - (kap*kp*(r0 + r1)*sp2)/((i-1)*h) + (kap*kp*(r1 + r0)*P[j, i])/((i-1)*h) + (kap*kp*r0*sp3)/((i-1)*h) - (kap*kp*r0*P[j, i-1])/((i-1)*h)
          
            alpha[i+1] = C / (B - A*alpha[i])       
            beta[i+1] = (F + A*beta[i]) / (B - A*alpha[i])              
        
        for i in range(N-2, 3, -1):    
            P[j+1, i] = alpha[i+1] * P[j+1, i+1] + beta[i+1]     
        
        for i in range(2, N-1):
            U[j+1, i] = abs((k0*(P[j+1, i+1] - P[j+1, i] + kp2*(sp1 + P[j+1, i+1] - P[j, i+1] - sp2 - P[j+1, i] + P[j, i])) / (myu0*h*(1 + kv1))) - (kv1*st - kv1*U[j, i]) / (1 + kv1))  # tezlik
        
        U[j+1, -1] = U[j+1, -2]   # tezlik
        print(j)
    
    # U[T-2, :]   # tezlik
    print(P[T-2, :])
    # plt.plot(np.arange(N)*h, abs(U[T-1, :]))  # tezlik 
    plt.plot(np.arange(N)*h, P[T-1, :])
    plt.show()

if __name__ == "__main__":
    filtration()