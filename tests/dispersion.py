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

N_t = 1000
N_z = 1000

# Physical parameters

# Source
T = 100*ps
λ = 1550 * nm
P0 = 1 * mW

# Fiber

α = 0.046 / km
γ = 1.1 / W / km
L = 5 * km
D = -17 * ps / nm /km

β2 = - D*λ**2/(2*np.pi*c) # dispersion
τ0 = 5*ps # pulse FWHM

def gaussian_pulse(t):
    return np.sqrt(P0)*np.exp(-((t-T/2)/(2*τ0))**2)


fib = Fiber(L, α, β2, γ) # create fiber
sim = SegmentSimulation(fib, N_z, N_t, gaussian_pulse, T) # simulate on the fiber portion
t, U = sim.run() # perform simulation


Pmatrix = np.abs(U)**2/mW # compute optical power matrix

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
plt.plot(t/ps,Pmatrix[0,:], label=r'Pulse at z={:.2f} km'.format(0))
plt.plot(t/ps,Pmatrix[-1,:], label=r'Pulse at z={:.2f} km'.format(L/km))
plt.grid()
plt.legend()
plt.ylabel(r'Optical power [mW]')
plt.xlabel(r'Local time [ns]')
plt.tight_layout()
plt.show()

