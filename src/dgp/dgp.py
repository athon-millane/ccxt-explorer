import numpy as np
import matplotlib.pyplot as plt


class Stationary:
    
    def __init__(self, N=10000, sigmaX = 0.05, sigmaEta = 0.1, theta = 0.1, mu = 100.):
        
        return None
        
    def generate(self, N=None):
        
        self.N = N if N is not None else self.N

        X = []
        Y = []
        epsilon = [self.mu]
        for t in range(self.N):
            if len(X) == 0:
                X.append(np.random.normal(10., self.sigmaX))
            else:
                X.append(X[-1] + np.random.normal(0., self.sigmaX))

            epsilon.append(epsilon[-1] + self.theta * (self.mu - epsilon[-1]) 
                                       + np.random.normal(0., self.sigmaEta))

            Y.append(X[-1] + epsilon[-1])

        X = np.array(X)
        Y = np.array(Y)

        return X, Y


class NonStationary:
    
    def __init__(self, N=10000, sigmaX=0.05, sigmaEta=0.1, 
                 theta=0.05, muX=10, mu=100, jump=3, cauchy=0.01):
        
        self.N = N
        self.sigmaX = sigmaX
        self.sigmaEta = sigmaEta
        self.theta = theta
        self.muX = muX
        self.mu = mu
        self.jump = jump
        self.cauchy = cauchy


    def generateDGP(self, N=None):
        
        self.N = N if N is not None else self.N
        X = []
        Y = []
        epsilon = [self.mu]
        for t in range(self.N):
            if t % 200 == 0:
                self.mu += np.random.standard_cauchy() * self.cauchy

            if np.random.uniform(0, 1) >= 0.995:
                self.mu += np.random.uniform(-self.jump, self.jump)

            if len(X) == 0:
                X.append(np.random.normal(self.muX, self.sigmaX))
            else:
                X.append(X[-1] + np.random.normal(0., self.sigmaX))

            epsilon.append(epsilon[-1] + self.theta * (self.mu - epsilon[-1]) 
                                       + np.random.normal(0., self.sigmaEta))

            Y.append(X[-1] + epsilon[-1])

        X = np.array(X)
        Y = np.array(Y)

        return X, Y