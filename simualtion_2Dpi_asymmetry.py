import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from numba import jit
#importing modules 
N= 53 #dimension of Pr matrix

@jit(nopython=True)
def Gfunc(qs,qi,e):
    p = 2 #sigma_p
    c = 16.#sigma_c
    s = 2.5 * c #sigma_s
    ci = abs(1 / (1 - e)) #cs=1
    return (30./c)*np.exp(- ((p**2+c**2)*(qs**2+qi**2)+2*qs*qi*c**2)/(2*p**2*c**2))*np.sinc(((2+e)*0.5*abs(qs-ci*qi)**2 + e*0.5*abs(qs+ci*qi)**2)/(np.pi*s**2)) #simplified collected JTMA

for e in np.arange(0,1,0.1): #loop for different values of epsilon
    G = np.identity(N) #initialising G matrix
    for i in range (N): #loop for a_s
        for j in range(N): #loop for a_i
            a_s= i-(N-1)/2
            a_i= j-(N-1)/2
            G[i][j]= Gfunc(a_s,a_i,e)

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), G)
    plt.colorbar()
    plt.title('JTMA $\epsilon$ = %.2f' % e)
    plt.show() #visualising collected JTMA

    '''
    def phi(q, a):
        if q < a:
            return 1.
        else:
            return -1.
    def Pr(qs,qi,e,a_s,a_i):
        return phi(qs,a_s)*phi(qi,a_i)*Gfunc(qs,qi,e)
    '''
    #Above two functions are combined to give below single function
    def Pr(qs, qi, e, a_s, a_i):
        if qs>a_s and qi>a_i or qs<a_s and qi<a_i:
            return Gfunc(qs, qi, e)
        else:
            return -Gfunc(qs,qi,e)

    Pr_M= np.identity(N) #initialising Pr matrix
    for i in range(N):
        for j in range(N):
            a_i = i - (N - 1) / 2
            a_s = j - (N - 1) / 2
            #I1= sci.dblquad(Pr,-np.inf,np.inf,-np.inf,np.inf,args=[e,a_s   ,a_i])[0] this integral can be broken into below 4 integrals
            I1=sci.dblquad(Pr,-np.inf,0,-np.inf,0,args=[e,a_s,a_i])[0]
            I2=sci.dblquad(Pr,-np.inf,0,0,np.inf,args=[e,a_s,a_i])[0]
            I3=sci.dblquad(Pr,0,np.inf,-np.inf,0,args=[e,a_s,a_i])[0]
            I4 = sci.dblquad(Pr,0,np.inf,0,np.inf,args=[e,a_s,a_i])[0]
            Pr_M[i][j] = abs(I1+I2+I3+I4) ** 2
            print(i, j, Pr_M[i][j])

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr_M)
    plt.colorbar()
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('Pr($a_s$, $a_i$) $\epsilon$ = %.2f' % e)
    plt.show() #visualising Pr matrix
