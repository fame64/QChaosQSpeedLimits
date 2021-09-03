import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.cm as cm
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size']=17
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import numba
from numba.types import *
from numba import prange
#n_t=4*24
t_arr=np.arange(0, 200, 1)
#t_arr=t_arr/50
#print(t_arr)
n_t=len(t_arr)
S=10
dim=int(2*S+1)
basis = np.identity(dim)
i=complex(0,1)
#Definition of the ladder operators
@numba.jit(numba.types.Tuple((float64, float64[:]))(float64[:], float64), nopython=True, parallel=False)
def S_p(ket, S):
    site= np.argmax(ket)
    m=site-S
    ket_out=np.zeros(dim)
    if site== 2*S:
        coef=0
        return (coef, ket_out)
    else:
        coef=np.sqrt(S*(S+1)-m*(m+1))
        ket_out[site+1]= 1
        return (coef, ket_out)
@numba.jit(numba.types.Tuple((float64, float64[:]))(float64[:], float64), nopython=True, parallel=False)
def S_m(ket, S):
    site= np.argmax(ket)
    m=site-S
    ket_out=np.zeros(dim)
    if site== 0:
        coef=0
        return (coef, ket_out)
    else:
        coef= np.sqrt(S*(S+1)-m*(m-1))
        ket_out[site-1]= 1
        return (coef, ket_out)
@numba.jit(float64(float64[:], float64[:]), nopython=True, parallel=False)
def braket(bra, ket):
    if np.all(bra== ket):
        return 1
    else:
        return 0
@numba.jit(complex128[:,:](float64[:, :], float64), nopython=True, parallel=False)
def Compute_Sx(basis, S):
    dim=int(2*S+1)
    Sx=np.zeros((dim, dim), dtype=complex128)
    for j in range(dim):
        for k in range(dim):
            ket=basis[j, :]
            bra=basis[k, :]
            coef_p, S_p_ket=S_p(ket, S)
            coef_m, S_m_ket=S_m(ket, S)
            result_p=braket(bra, S_p_ket)
            result_m=braket(bra, S_m_ket)
            Sx[j, k]=(coef_p*result_p+ coef_m*result_m)/2
    return Sx

@numba.jit(complex128[:,:](float64[:, :], float64), nopython=True, parallel=False)
def Compute_Sy(basis, S):
    dim=int(2*S+1)
    Sy=np.zeros((dim, dim), dtype=complex128)
    i=complex(0,1)
    for j in range(dim):
        for k in range(dim):
            ket=basis[j, :]
            bra=basis[k, :]
            coef_p, S_p_ket=S_p(ket, S)
            coef_m, S_m_ket=S_m(ket, S)
            result_p=braket(bra, S_p_ket)
            result_m=braket(bra, S_m_ket)
            Sy[j, k]=-i*(coef_p*result_p- coef_m*result_m)/2
    return Sy

@numba.jit(complex128[:,:](float64), nopython=True, parallel=False)
def Compute_Sz(S):
    dim=int(2*S+1)
    Sz=np.zeros((dim, dim), dtype=complex128)
    for j in range(dim):
        m=j-S
        Sz[dim-1-j, dim-1-j]=m
    return Sz

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@numba.jit(float64(int64), nopython=True, parallel=False)
def fast_log_factorial(n):
    if n<20:
        return np.log(LOOKUP_TABLE[n])
    else:
        #stirling approx
        return n*np.log(n)-n+ 0.5*np.log(2*np.pi*n)
        #return n*np.log(n)-n+ 0.5*np.log(2*np.pi*n)+ 1/(12*n)-1/(360*n**3)
#Take into account that z is complex
@numba.jit(complex128[:](float64, float64, float64), nopython=True, parallel=False)
def define_zeta(S, theta, phi):
    z=np.exp(i*phi)*np.tan(theta/2)
    dim=int(2*S+1)
    ket_z=np.zeros(dim, dtype=complex128)
    for k in range(dim):
        m=k-S
        aux1=fast_log_factorial(int(2*S)) #aux1= log((2S)!)
        aux2=fast_log_factorial(int(S+m)) #aux2= log((S+m)!)
        aux3=fast_log_factorial(int(S-m)) #aux3= log((S-m)!)
        #ket_z[j]=np.sqrt(aux1/(aux2*aux3))*z**(S+m)/(1 + (np.abs(z))**2)**S
        #To deal with big numbers we introduce the log
        #of the factorial, compute the division of them and afterwards we exponentiate
        ket_z[k]=np.exp((aux1-aux2-aux3)/2)*z**(S+m)/((1 + (np.abs(z))**2)**S)
    norm= np.sqrt(np.conjugate(ket_z)@ket_z)
    return ket_z/norm
@numba.jit(complex128[:,:](complex128[:], complex128[:]), nopython=True, parallel=False)
def ketbra(ket, bra):
    dim=len(ket)
    res=np.zeros((dim, dim), dtype=complex128)
    for j in range(dim):
        for k in range(dim):
            res[j,k]=ket[j]*np.conjugate(bra[k])
    return res
def define_zeta_alt(theta, phi, dim, Sx, Sy, i):
    g=np.exp(i*phi)*np.tan(theta/2)
    ket_z=np.zeros(dim, dtype=complex)
    ket_z[0]=1
    S_m=Sx-i*Sy
    S=(dim-1)/2
    aux=expm(g*S_m)
    ket_z=1/(1 + np.conjugate(g)*g)**S*aux@ket_z
    #print(np.conjugate(ket_z)@ket_z)
    return ket_z


#@numba.njit(numba.types.Tuple((complex128, complex128, complex128))(complex128[:,:],complex128[:,:], complex128[:,:], complex128[:] ))
def computeXYZ(Sx, Sy, Sz, psi):
    X_=np.conjugate(psi)@Sx@psi
    Y_=np.conjugate(psi)@Sy@psi
    Z_=np.conjugate(psi)@Sz@psi
    return (X_, Y_, Z_)

def computeQ(rho, ket_th_ph, nt, nphi):
    Q=np.zeros((nt, nphi))
    for j in range(nt):
        for k in range(nphi):
            Q[j, k]=np.real(np.conj(ket_th_ph[j,k, :])@rho@ket_th_ph[j,k,:]*(2*S+1)/(4*np.pi))
    return Q

def evEqns(X, Y, Z, p, k,i, S):
    aux=0.5*(X*np.cos(p)+Z*np.sin(p)+ i*Y)@expm(i*k/S*(Z*np.cos(p)-X*np.sin(p)+ 0.5))
    #X_n=0.5*(X*np.cos(p)+Z*np.sin(p)+ i*Sy)@expm(i*k/S*(Z*np.cos(p)-X*np.sin(p)+ 0.5))+0.5*(X*np.cos(p)+Z*np.sin(p)- i*Sy)@expm(-i*k/S*(Z*np.cos(p)-X*np.sin(p)+ 0.5))
    X_n=aux+np.transpose(np.conj(aux))
    Y_n=(aux+np.transpose(np.conj(aux)))/i
    Z_n=Z*np.cos(p)-X*np.sin(p)
    return (X_n, Y_n, Z_n)

Sx=np.zeros((dim, dim), dtype=complex)
Sy=np.zeros((dim, dim), dtype=complex)
Sz=np.zeros((dim, dim), dtype=complex)
Sx=Compute_Sx(basis, S)
Sy=Compute_Sy(basis, S) #I've checked that for S=1/2, 1, 3/2 the results are the expected ones
Sz=Compute_Sz(S)
#S2=Sx@Sx+ Sy@Sy+Sz@Sz
jm=np.zeros(dim)
"""m=9
jm[m]=1
print(S2@jm/(S*(S+1)))"""

p_free=np.pi/2
k_kick=5
i=complex(0,1)
#dt=t_arr[1]-t_arr[0]
#Udt=expm(-i*p*Sx*dt)
#Ukick=expm(i*k*Sz@Sz/(2*S+1))
U=expm(- i*k_kick*Sz@Sz/(2*S))@expm(-i*p_free*Sy)
Uinv=expm(i*p_free*Sy)@expm(i*k_kick*Sz@Sz/(2*S))
#print(S2@U-U@S2)
theta_coh=1
phi_coh=2

psi0=define_zeta(S, theta_coh, phi_coh)
#psi0=define_zeta_alt(theta_coh, phi_coh, dim, Sx, Sy, i)
rho0=ketbra(psi0, psi0)
rho_t=rho0
n_theta=50
n_phi=2*n_theta
epsilon=1e-5
theta_arr=np.linspace(0, np.pi-epsilon, n_theta)
phi_arr=np.linspace(0, 2*np.pi, n_phi)
Th, Ph=np.meshgrid(phi_arr, theta_arr)
coh_states=np.zeros((n_theta, n_phi, dim), dtype=complex)
Q_t=np.zeros((n_theta, n_phi, n_t))
for j in range(n_theta):
    for k in range(n_phi):
        coh_states[j,k,:]=define_zeta(S, theta_arr[j], phi_arr[k])
        #coh_states[j,k,:]=define_zeta_alt(theta_arr[j], phi_arr[k], dim, Sx, Sy, i)
for k in range(n_t):
    Q_t[:,:,k]=computeQ(rho_t, coh_states, n_theta, n_phi)
    rho_t=U@rho_t@Uinv

#print(Q_t)
"""plt.figure()
plt.contourf(Th, Ph, Q_t[:,:,39])
plt.show()"""
x_arr=np.zeros((n_theta, n_phi, n_t))
y_arr=np.zeros((n_theta, n_phi, n_t))
z_arr=np.zeros((n_theta, n_phi, n_t))
Rmax=0.2
for j in range(n_theta):
    for k in range(n_phi):
        for l in range(n_t):
            x_arr[j,k,l] = (1 + Rmax*Q_t[j,k,l]) * np.cos(Ph[j,k]) * np.sin(Th[j,k])
            y_arr[j,k,l] = (1 + Rmax*Q_t[j,k,l]) * np.sin(Ph[j,k]) * np.sin(Th[j,k])
            z_arr[j,k,l] = (1 + Rmax*Q_t[j,k,l]) * np.cos(Th[j,k])
# ANIMATION FUNCTION
frn =n_t
#fps=50
def func(num, dataSet, line, dataSet2, line2):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, :num])
    line.set_3d_properties(dataSet[2, :num])
    line2.set_data(dataSet2[0:2, :num])
    line2.set_3d_properties(dataSet2[2, :num])
    return line
# THE DATA POINTS

# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure(figsize=[5,5])
ax = Axes3D(fig)
"""u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.4)"""
def update_plot(frame_number, x_arr, y_arr, z_arr, Q_t, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x_arr[:,:,frame_number], y_arr[:,:,frame_number], z_arr[:,:,frame_number], facecolors=cm.plasma(Q_t[:,:,frame_number]/np.max(Q_t[:,:,frame_number])), cmap="plasma", alpha=0.4)

plot = [ax.plot_surface(x_arr[:,:,0], y_arr[:,:,0], z_arr[:,:,0], facecolors=cm.plasma(Q_t[:,:,0]/np.max(Q_t[:,:,0])), cmap="plasma", rstride=1, cstride=1, alpha=0.4)]
#ax.set_zlim(-1.5, 1.5)
ani = FuncAnimation(fig, update_plot, frn, fargs=(x_arr, y_arr, z_arr, Q_t, plot), interval=0.00001)
# NOTE: Can't pass empty arrays into 3d version of plot()
# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel(r'$Q(\theta, \phi) \cos \phi \sin \theta$')
ax.set_ylabel(r'$Q(\theta, \phi) \sin \phi \sin \theta$')
ax.set_zlabel(r'$Q(\theta, \phi) \cos \theta$')
ax.set_xlim(-1-Rmax, 1+Rmax)
ax.set_ylim(-1-Rmax, 1+Rmax)
ax.set_zlim(-1-Rmax, 1+Rmax)
#plot.colorbar()
#ax.set_title('Trajectory of electron for E vector along [120]')

# Creating the Animation object
#line_ani = FuncAnimation(fig, func, frames=n_t, fargs=(dataSet,line, dataSet2, line2), interval=50, blit=False)
#line_ani.save(r'AnimationNew.mp4')


plt.show()
"""f=0
fig = plt.figure(figsize=[5,5])
axes = Axes3D(fig)
plt.rcParams["figure.autolayout"] = True
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
axes.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.3)
#axes.scatter(X_t/S, Y_t/S, Z_t/S, s=2, color='black')
def update_plot(frame):
    global f
    plt.clf()
    axes.scatter(X_t[0:f]/S, Y_t[0:f]/S, Z_t[0:f]/S, s=2, color='black')
    axes.scatter(X_t[f]/S, Y_t[f]/S, Z_t[f]/S, s=25, color='red')
    #axes.quiver(0,0,0,X_t[f]/S, Y_t[f]/S, Z_t[f]/S, color='red')
    f=f+1
    #plt.scatter(y[-1], x[-1])
    #plt.savefig()
animation=FuncAnimation(fig, update_plot, interval=1)
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
axes.set_zlim(-1, 1)
plt.show()"""
