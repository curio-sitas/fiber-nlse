import numpy as np
from tqdm import tqdm


class Fiber:
    def __init__(self, L:int, α:float, β2: float, γ:float) -> None:

        self.α = α
        self.L = L
        self.β2 = β2
        self.γ = γ

class SegmentSimulation:
    def __init__(self, fiber: Fiber, N_l, N_t, u0, T0:float, loader=True) -> None:

        self.loader = loader
        self.fiber = fiber
        self.N_l = N_l
        self.N_t = N_t
        self.T0 = T0
        self.dt = T0/N_t
        self.U = np.zeros((N_l, N_t), dtype=np.complex)
        self.t = np.linspace(0,1, N_t)*T0
        self.U[0,:] = u0(self.t)
        self.ν = np.fft.fftfreq(len(self.t), d=self.dt)

    def load_fiber(self, fiber):
        self.fiber = fiber
    def NL(self,u):
        return 1j * np.abs(u) ** 2 * self.fiber.γ
    def DISP(self):
        return -self.fiber.β2*0.5*1j*(2*np.pi*self.ν)**2

    def nlse_step(self, u):

        α = self.fiber.α
        dl = self.fiber.L/self.N_l
        ui = np.fft.ifft(np.exp(0.5 * dl * (self.DISP()-0.5*α)) * np.fft.fft(u))  # disp on half step
        ui = np.exp(dl * self.NL(ui)) * ui  # full step NL
        ui = np.fft.ifft(np.exp(0.5 * dl * (self.DISP()-0.5*α)) * np.fft.fft(ui))  # disp on half step
        return ui

    def run(self):
        if(self.loader):
            for i in tqdm(range(1,self.N_l)):
                self.U[i] = self.nlse_step(self.U[i-1])
        else:
            for i in tqdm(range(1,self.N_l)):
                self.U[i] = self.nlse_step(self.U[i-1])
        return (self.t, self.U)


#TODO

"""

* Implement multi-segment simulation
* Add self steepening
* Add raman diffusion
* Add Autocorrelation
* Add visualizations
* Implement RK4 integration
* Implement multi-threading

"""