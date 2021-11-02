import math
import numpy as np
import pyfftw
from tqdm import tqdm


# Reference : https://doi.org/10.1364/OE.18.008261

# Define space-time grid

# Define β function over z and nth order


class FiberNLSE:

    #def __init__(self):
        # make default inits


    def initDimensions(self, DT, Nt, L, Nl):
        self.DT = DT
        self.L = L
        self.Nt = Nt
        self.Nl = Nl
        self.dt = self.DT / Nt
        self.dl = L / Nl

        self.F = np.fft.fftfreq(Nt, d=self.dt)

    def initPrepulse(self, f):
        self.T = 0.5*np.linspace(-self.DT, self.DT, self.Nt)
        self.A = np.zeros((self.Nl, self.Nt), dtype=complex)
        self.A[0, :] = f(self.T)

    def initPrepulseArray(self, Ap):
        self.T = 0.5*np.linspace(-self.DT, self.DT, self.Nt)
        self.A = np.zeros((self.Nl, self.Nt), dtype=complex)
        self.A[0, :] = Ap

    def initNonLinearity(self, γ):
        self.γ = γ

    def initLosses(self, α):
        self.α = α

    # TODO: dos not work
    # TODO: create a dispersion model class
    def initUniformDispersion(self, β):
        self.β = β  # β must start at the second order β2
        self.D = np.sum([1j**(2*i+1)*(2 * np.pi * self.F) ** (i + 2) * float(self.β[i]) / math.factorial(i + 1) for i in range(len(self.β))])


    def initDispersion2(self, β2):
        self.β = β2  # β must start at the second order β2
        self.D = -self.β*0.5*1j*(2*np.pi*self.F)**2

    def calculateN(self, A):
        return 1j * np.abs(A) ** 2 * self.γ

    def simulate(self):
        # return A, A_fft, F, T, L
        for i in tqdm(range(1, self.Nl)):
            self.A[i] = self.step_forward(self.A[i - 1])
        return self.A

    # TODO: RK4
    def step_forward(self, A):

        h = self.dl
        x = pyfftw.empty_aligned(self.N, dtype="complex128")
        X = pyfftw.empty_aligned(self.N, dtype="complex128")
        plan_forward = pyfftw.FFTW(x, X)
        plan_inverse = pyfftw.FFTW(X, x, direction="FFTW_BACKWARD")
        
        N = self.calculateN(A)  # Nonlinear operator
        Ai = np.fft.ifft(np.exp(0.5 * h * (self.D-0.5*self.α)) * np.fft.fft(A))  # disp on half step
        Ai = np.exp(h * N) * Ai  # full step NL
        Ai = np.fft.ifft(np.exp(0.5 * h * (self.D-0.5*self.α)) * np.fft.fft(Ai))  # disp on half step
        return Ai
