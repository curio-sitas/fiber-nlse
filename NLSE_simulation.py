

# %%
import pylab as plt
import numpy as np
import scipy
from tqdm import tqdm


# %%
def step_forward(u_last):
    NL = 1j*np.abs(u_last)**2*N**2  # Nonlinear operator
    ui = np.fft.ifft(np.exp(0.5 * dζ * DISP) * np.fft.fft(u_last))  # disp on half step
    ui = np.exp(dζ * NL) * ui  # full step NL
    ui = np.fft.ifft(np.exp(0.5 * dζ * DISP) * np.fft.fft(ui))  # disp on half step
    return ui


# %%
# scales

GHz =   1e9
ps  =   1e-12
ns  =   1e-9
km  =   1e3
nm  =   1e-9
mW  =   1e-3
W   =   1

# Constants

c = 3e8

# Simulation Parameters

Lmax = 5*km
Tmax = 200*ps

N_z = 2000
N_t = 2000

# Physical Parameters

f0 = 10 * GHz # envelope frequency
λ = 1550 * nm # source wavelength
γ = 1.1 /W/km # nonlinear factor
D = - 17 * ps/nm/km # dispersion factor
α =  0 #0.046 / km # Fiber losses
β2 = - D*λ**2/(2*np.pi*c) # dispersion
T0 = 1/f0


P0 = 500 * mW # envelope power
P0 = β2/(γ*T0**2) # ideal soliton power (N²=1)

# Vectors

t = Tmax*np.linspace(0,1, N_t) # time vector
z = Lmax*np.linspace(0,1, N_z) # distance vector

# Intermediate calculus

L_D = T0**2/np.abs(β2) # Dispersion length
L_N = 1/(P0*γ) # Nonlinear length
N = np.sqrt(L_D/L_N) # Soliton order

# Non-dimensionalization

ζ = z/L_D # adimension distance
τ = t/T0 # adimension local time

ζmax = Lmax/L_D
τmax = max(τ)

# steps
dτ = abs(τ[1]-τ[0])
dζ = abs(ζ[1]-ζ[0])

# adimension frequency
ν = (np.fft.fftfreq(N_t, d=dτ)) 

sgn = -1 # anomalous dispersion +1 if normal dispersion
DISP = sgn*0.5*(1j*(2*np.pi*ν)**2) - 0.5*α*L_D # dispersion operator (with losses)



u = np.zeros((N_z, N_t), dtype=complex) # time/frequency matrix
δ = np.pi*1.6 # phase modulation factor
θ = np.pi # phase shift constant
κ = 0.1 # phase modulation input factor
up = np.cos(2*np.pi*τ) # envelope
up = up*np.exp(1j*(δ*np.cos(θ+κ*up))) # phase
u[0,:] = up # cosine-squared enveloppe at f0

u[0, :] = 1/np.cosh(τ-0.5*τmax) # soliton initial pulse


# %%
for i in tqdm(range(1, len(ζ))):
    u[i] = step_forward(u[i - 1])


# %%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
U = np.abs(u)**2*P0
plt.imshow(U, extent=[0, Tmax/ps, Lmax/km, 0], aspect=20)
cm = plt.colorbar()
plt.ylabel(r"Distance [km]")
plt.xlabel(r"Local time [ps]")


plt.subplot(1,2,2)
plt.plot(t/ps,np.abs(U[0,:])/mW, lw=1, ls='--', color='black', label=r'Initial z=0')
plt.plot(t/ps,np.abs(U[-1,:])/mW, lw=1, color='red', label=r'Final z={} km'.format(Lmax/km))
plt.grid()
plt.legend()
plt.ylabel(r"Power [mW]")
plt.xlabel(r"Local time [ps]")

plt.tight_layout()
plt.show()


# %%

T, Z = np.meshgrid(t, z)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T/ps, Z/km, U)
plt.show()


# %%
