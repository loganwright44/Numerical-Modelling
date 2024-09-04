# Ising model to compute specific heat of a square lattice as a function of temperature
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, uniform, random

class SquareLattice:
    def __init__(self, N : int, J : float = 1.0):
        self.N = N
        self.J = J
        self.lattice = np.zeros((N, N))
        self.FillLattice()
        self.kB = 1
        # Parameters for the characterization of the overall lattice
        self.E = 0.0
        self.M = 0.0
    
    def FillLattice(self):
        # Assigns random spin states to the lattice points on the grid
        for i in range(self.N):
            for j in range(self.N):
                self.lattice[i][j] = -1.0 if uniform(0, 1) <= 0.5 else 1.0
        
        return None
    
    def MetropolisAlgorithm(self, T : float):
        self.E = 0.0
        
        # Make an additional function to handle both direct and probabilistic spin flips on the state [i, j]
        def UpdateLattice(dE, i, j, randomNumber):
            if randomNumber > 0.5:
                self.E -= dE
                return None
            
            # Apply the metropolis algorithm
            if dE > 0:
                if uniform(0, 1) <= np.exp(-dE / (self.kB * T)):
                    self.lattice[i, j] *= -1.0
                    self.E += dE
                else:
                    self.E -= dE
            else:
                self.lattice[i, j] *= -1.0
                self.E += dE
                
            return None
        
        # Since the energy calculated requires a permutation of the spins states of all neartest neighbors, a separate function seems a more clear solution
        def SpinPairSummation(coreSpin : float, neighbors : list | tuple):
            return coreSpin * sum(neighbors)
        
        # Copy the lattice to a temporary storage location
        latticeTemp = np.copy(self.lattice)
        # Iterate through the 2 indeces of the lattice and make changes to the REAL lattice while looking and the
        # temporary lattice copy for reference of previous spin states for calculations
        for i in range(1, self.N - 1):
            # Deal with the boundaries on the edges...
            E1 = -self.J * SpinPairSummation(latticeTemp[i, 0], (latticeTemp[i - 1, 0], latticeTemp[i, 1], latticeTemp[i + 1, 0]))
            E2 = -self.J * SpinPairSummation(-latticeTemp[i, 0], (latticeTemp[i - 1, 0], latticeTemp[i, 1], latticeTemp[i + 1, 0]))
            UpdateLattice(E2 - E1, i, 0, uniform(0, 1))
            E1 = -self.J * SpinPairSummation(latticeTemp[0, i], (latticeTemp[0, i + 1], latticeTemp[1, i], latticeTemp[0, i - 1]))
            E2 = -self.J * SpinPairSummation(-latticeTemp[0, i], (latticeTemp[0, i + 1], latticeTemp[1, i], latticeTemp[0, i - 1]))
            UpdateLattice(E2 - E1, 0, i, uniform(0, 1))
            E1 = -self.J * SpinPairSummation(latticeTemp[i, -1], (latticeTemp[i + 1, -1], latticeTemp[i, -2], latticeTemp[i - 1, -1]))
            E2 = -self.J * SpinPairSummation(-latticeTemp[i, -1], (latticeTemp[i + 1, -1], latticeTemp[i, -2], latticeTemp[i - 1, -1]))
            UpdateLattice(E2 - E1, i, -1, uniform(0, 1))
            E1 = -self.J * SpinPairSummation(latticeTemp[-1, i], (latticeTemp[-1, i - 1], latticeTemp[-2, i], latticeTemp[-1, i + 1]))
            E2 = -self.J * SpinPairSummation(-latticeTemp[-1, i], (latticeTemp[-1, i - 1], latticeTemp[-2, i], latticeTemp[-1, i + 1]))
            UpdateLattice(E2 - E1, -1, i, uniform(0, 1))
            
            for j in range(1, self.N - 1):
                # Ising model of the energy using nearest neighbor approximations
                E1 = -self.J * SpinPairSummation(latticeTemp[i, j], (latticeTemp[i, j + 1], latticeTemp[i - 1, j], latticeTemp[i, j - 1], latticeTemp[i + 1, j]))
                E2 = -self.J * SpinPairSummation(-latticeTemp[i, j], (latticeTemp[i, j + 1], latticeTemp[i - 1, j], latticeTemp[i, j - 1], latticeTemp[i + 1, j]))
                UpdateLattice(E2 - E1, i, j, uniform(0, 1))
        
        # And finally, deal with the corners
        E1 = -self.J * SpinPairSummation(latticeTemp[0, 0], (latticeTemp[0, 1], latticeTemp[1, 0]))
        E2 = -self.J * SpinPairSummation(-latticeTemp[0, 0], (latticeTemp[0, 1], latticeTemp[1, 0]))
        UpdateLattice(E2 - E1, 0, 0, uniform(0, 1))
        E1 = -self.J * SpinPairSummation(latticeTemp[0, -1], (latticeTemp[1, -1], latticeTemp[0, -2]))
        E2 = -self.J * SpinPairSummation(-latticeTemp[0, -1], (latticeTemp[1, -1], latticeTemp[0, -2]))
        UpdateLattice(E2 - E1, 0, -1, uniform(0, 1))
        E1 = -self.J * SpinPairSummation(latticeTemp[-1, -1], (latticeTemp[-1, -2], latticeTemp[-2, -1]))
        E2 = -self.J * SpinPairSummation(-latticeTemp[-1, -1], (latticeTemp[-1, -2], latticeTemp[-2, -1]))
        UpdateLattice(E2 - E1, -1, -1, uniform(0, 1))
        E1 = -self.J * SpinPairSummation(latticeTemp[-1, 0], (latticeTemp[-2, 0], latticeTemp[-1, 1]))
        E2 = -self.J * SpinPairSummation(-latticeTemp[-1, 0], (latticeTemp[-2, 0], latticeTemp[-1, 1]))
        UpdateLattice(E2 - E1, -1, 0, uniform(0, 1))
        
        self.M = np.sum(self.lattice) / (self.N * self.N)
        self.E /= (self.N * self.N)
        return self.E, self.M
    
    def Animate(self, nIters : int, T : float):
        e = []
        m = []
        
        for _ in range(nIters):
            EM = self.MetropolisAlgorithm(T = T)
            plt.imshow(self.lattice, cmap = 'cool')
            plt.draw()
            plt.pause(0.1)
            plt.clf()
            e.append(EM[0])
            m.append(EM[1])
        
        plt.close()
        return e, m

lattice = SquareLattice(N = 10, J = 1)
E, M = lattice.Animate(nIters = 100, T = 1.2e4)
N = list(range(1, 101))