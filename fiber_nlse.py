import numpy as np

# Reference : https://doi.org/10.1364/OE.18.008261

# Define space-time grid

# Define β function over z and nth order

# Define N operator

# Define D operator

# One step

# N step

class FiberNLSE():
    def __init__(self):
        return

    def initDimensions(self, T, Nt, L, Nl):
        self.T = T
        self.L = L
        self.Nt = Nt
        self.Nl = Nl
        self.dt = T/Nt
        self.dl = L/Nl

        self.F = np.fft.fftfreq(Nt, d=self.dt)

    def initPrepulse(self, f):
        self.A = np.zeros(self.Nl, self.Nt)
        self.A[0,:] = f(self.T)

    def initNonLinearity(self, γ):
        self.γ = γ

    def initLosses(self, α):
        self.α = α

    # TODO: create a dispersion model class
    def initUniformDispersion(self, β):
        self.β = β # β must start at the second order β2
        self.D = np.sum([(2*np.pi*1j*self.F)**(i+1)*β[i] for i in range(self.β)])

    def calculateN(self, A):
        return np.abs(A)**2*self.γ

    def simulate(self):
        # return A, A_fft, F, T, L
        for i in range(1,self.Nt):
            self.A[i] = self.step_forward(self.A[i-1])
        return
