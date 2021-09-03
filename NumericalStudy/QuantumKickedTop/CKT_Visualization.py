import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size']=17
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import numba
from numba.types import *
from numba import prange
#n_t=4*24
t_arr=np.arange(0, 100, 1)
#t_arr=t_arr/50
#print(t_arr)
n_t=len(t_arr)
#S=10
#dim=int(2*S+1)
#basis = np.identity(dim)
#Definition of the ladder operators


def evEqns(X, Y, Z, p, k,i):
    aux=(X*np.cos(p)+ Z*np.sin(p)+i*Y)*np.exp(i*k*(Z*np.cos(p)-X*np.sin(p)))
    X_n=np.real(aux)
    Y_n=np.imag(aux)
    Z_n=-X*np.sin(p)+Z*np.cos(p)
    return (X_n, Y_n, Z_n)

p=1
k=5
i=complex(0,1)

#print(S2@U-U@S2)
theta=np.pi/2
phi=0.5
psi0=np.asarray([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

eps=1e-4
theta2=theta+eps
phi2=phi+eps
psi02=np.asarray([np.sin(theta2)*np.cos(phi2),np.sin(theta2)*np.sin(phi2),np.cos(theta2)])
#psi02=define_zeta_alt(z2, dim, Sx, Sy, i)

#psi_j=psi0
X_t=np.zeros(n_t)
Y_t=np.zeros(n_t)
Z_t=np.zeros(n_t)
#psi_j2=psi02
X_t2=np.zeros(n_t)
Y_t2=np.zeros(n_t)
Z_t2=np.zeros(n_t)

X_t[0]=psi0[0]
Y_t[0]=psi0[1]
Z_t[0]=psi0[2]
X_t2[0]=psi02[0]
Y_t2[0]=psi02[1]
Z_t2[0]=psi02[2]
for j in range(1, n_t):
    X_t[j], Y_t[j], Z_t[j]=evEqns(X_t[j-1], Y_t[j-1], Z_t[j-1], p, k, i)
    X_t2[j], Y_t2[j], Z_t2[j]=evEqns(X_t2[j-1], Y_t2[j-1], Z_t2[j-1], p, k, i)

#fig, (ax, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="3d"))
fig=plt.figure(figsize=[15, 7])
ax=fig.add_subplot(1, 2, 1, projection="3d")
ax2=fig.add_subplot(1, 2, 2)
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.4)
def get_arrow(u, v, w):
    x = 0
    y = 0
    z = 0
    #u = np.sin(2*theta)
    #v = np.sin(3*theta)
    #w = np.cos(3*theta)
    return x,y,z,u,v,w

quiver = ax.quiver(*get_arrow(X_t[0], Y_t[0], Z_t[0]), color='red')
quiver2 = ax.quiver(*get_arrow(X_t2[0], Y_t2[0], Z_t2[0]), color='blue')
scat=ax.scatter(X_t[0], Y_t[0], Z_t[0], color='red')
scat2=ax.scatter(X_t2[0], Y_t2[0], Z_t2[0], color='blue')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
d_plot=ax2.scatter(t_arr[0], np.sqrt((X_t[0]-X_t2[0])**2+(Y_t[0]-Y_t2[0])**2 + (Z_t[0]-Z_t2[0])**2), color='green')
ax2.set_yscale('log')
ax2.set_ylabel(r"$|\mathbf{r}_1(t)-\mathbf{r}_2(t)|$")
ax2.set_xlabel("$t$")
ax2.set_xlim(0, 100)
ax2.set_ylim(1e-4, 3)
#plt.savefig("frame0.png")
def update(t):
    global quiver
    global quiver2
    global scat2
    global scat
    global d_plot
    quiver.remove()
    quiver2.remove()
    scat2.remove()
    scat.remove()
    d_plot.remove()
    quiver = ax.quiver(*get_arrow(X_t[t], Y_t[t], Z_t[t]), color='red')
    quiver2 = ax.quiver(*get_arrow(X_t2[t], Y_t2[t], Z_t2[t]), color='blue')
    scat=ax.scatter(X_t[0:t], Y_t[0:t], Z_t[0:t], color='red')
    scat2=ax.scatter(X_t2[0:t], Y_t2[0:t], Z_t2[0:t], color='blue')
    d_plot=ax2.scatter(t_arr[0:t], np.sqrt((X_t[0:t]-X_t2[0:t])**2+(Y_t[0:t]-Y_t2[0:t])**2 + (Z_t[0:t]-Z_t2[0:t])**2), color='green')
    #plt.savefig("frame"+str(t)+".png")
ani = FuncAnimation(fig, update, frames=t_arr, interval=50)

plt.show()
#plt.clf()
