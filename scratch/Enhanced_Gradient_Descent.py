import numpy as np

class SGD():
    def __init__(self, eta=1e-3):
        self.eta = eta
        assert self.eta > 0 and isinstance(self.eta, (int, float))
    
    def Delta(self, dg):
        return -self.eta * dg

class Adagrad():
    def __init__(self, eta=1e-3, epsilon=1e-8):
        self.eta = eta
        self.epsilon = epsilon
        self.past_g = 0
        assert self.eta > 0 and isinstance(self.eta, (int, float))
        assert 0 < self.epsilon < 1 and isinstance(self.epsilon, float)
    
    def Delta(self, dg):
        self.past_g += dg**2
        return -self.eta * dg / (self.past_g**(1/2) + self.epsilon)

class RMSprop():
    def __init__(self, eta=1e-3, gamma=0.9, epsilon=1e-8):
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.v = 0
        assert self.eta > 0 and isinstance(self.eta, (int, float))
        assert 0 < self.gamma < 1 and isinstance(self.gamma, float)
        assert 0 < self.epsilon < 1 and isinstance(self.epsilon, float)
    
    def Delta(self, dg):
        self.v = self.gamma * self.v + (1 - self.gamma) * dg * dg
        return -self.eta * dg / (self.v**(1/2) + self.epsilon)

class Momentum():
    def __init__(self, eta=1e-3, gamma=0.9):
        self.eta = eta
        self.gamma = gamma
        self.m = 0
        assert self.eta > 0 and isinstance(self.eta, (int, float))
        assert 0 < self.gamma < 1 and isinstance(self.gamma, float)
    
    def Delta(self, dg):
        self.m = self.gamma * self.m - (1 - self.gamma) * dg
        return self.eta * self.m

class Adam():
    def __init__(self, eta=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m, self.v = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 0
        assert self.eta > 0 and isinstance(self.eta, (int, float))
        assert 0 < self.beta1 < 1 and isinstance(self.beta1, float)
        assert 0 < self.beta2 < 1 and isinstance(self.beta2, float)
        assert 0 < self.epsilon < 1 and isinstance(self.epsilon, float)
        
    def Delta(self, dg):
        self.t += 1
        self.m = self.beta1 * self.m - (1 - self.beta1) * dg
        self.v = self.beta2 * self.v + (1 - self.beta2) * dg**2
        hat_m = self.m/(1 - self.beta1**self.t)
        hat_v = self.v/(1 - self.beta2**self.t)
        return self.eta * hat_m / (hat_v**(1/2) + self.epsilon)