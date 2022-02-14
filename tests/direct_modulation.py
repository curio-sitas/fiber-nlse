#%%
import numpy as np
import sys
import pylab as plt

sys.path.append('../')

from fiber_nlse.fiber_nlse import *


# Physical units & constants

nm = 1e-9
ps = 1e-12
km = 1e3
mW = 1e-3
GHz = 1e9
Thz = 1e12
m = 1
W = 1
c = 3e8

# Simulation metrics

N_t = 2000
N_z = 1000

# Physical parameters

# Source
T = 500*ps
λ = 1550 * nm
P0 = 490 * mW
f0 = 10 * GHz

# Fiber

α = 0.046 / km
γ = 10.1 / W / km
γ2 = 1.1 / W / km
L2 = 5000 * m
L = 0 * m
D = -0.8 * ps / nm /km
D2 = - 20 * ps / nm / km
β2 = - D*λ**2/(2*np.pi*c) # dispersion
β2_2 = - D2*λ**2/(2*np.pi*c) # dispersion
τ0 = 10*ps # pulse FWHM

def gaussian_pulse(t):
    return np.sqrt(P0)*np.exp(-((t-T/2)/(2*τ0))**2)

def direct_modulation(t):
    return np.sqrt(P0)*np.cos(2*np.pi*f0*t)


fib = Fiber(L, α, β2, γ) # create fiber
sim = SegmentSimulation(fib, N_z, N_t, direct_modulation, T) # simulate on the fiber portion
t, U = sim.run() # perform simulation
Pmatrix = np.abs(U)**2
fib2 = Fiber(L2, α, β2_2, γ2) 
sim2 = SegmentSimulation(fib2, N_z, N_t, lambda x : U[-1,:], T) # simulate on the fiber portion
t, U2 = sim2.run() # perform simulation
Pmatrix = np.abs(np.vstack((U, U2)))**2/mW # compute optical power matrix

#%%
plt.figure()
plt.title(r'Pulse progagation with dipsersion')
plt.imshow(Pmatrix, aspect='auto', extent=[-T/2/ps, T/2/ps, L/km, 0])
plt.tight_layout()
plt.xlabel(r'Local time [ns]')
plt.ylabel(r'Distance [km]')
cb = plt.colorbar()
cb.set_label(r'Optical power [mW]')
plt.show()
# %%

plt.figure()
plt.title(r'Pulse propagation with dispersion')
plt.plot(t/ps,np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(U[0,:])))), label=r'Pulse at z={:.2f} km'.format(0))
plt.plot(t/ps,np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(U[-1,:])))), label=r'Pulse at z={:.2f} km'.format(L/km))
plt.grid()
plt.legend()
plt.ylabel(r'Optical phase [rad]')
plt.xlabel(r'Local time [ns]')
plt.tight_layout()
plt.show()


# %%
plt.plot(Pmatrix[-1,:])
plt.plot(Pmatrix[0,:])
plt.show()
# %%
