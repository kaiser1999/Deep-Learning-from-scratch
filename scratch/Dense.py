import numpy as np
import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))
from .Intermediate_Layers import Layer

class Dense(Layer):
    def __init__(self, d_ell):
        self.d_ell = d_ell
        self.b, self.W = None, None
        assert self.d_ell > 0 and isinstance(self.d_ell, int)
    
    def Param_init(self, input_shape):
        # ReLU -> He initialization
        n_input = input_shape[0]
        if self.W is None:
            self.W = np.random.normal(size=(self.d_ell, n_input), loc=0.0, 
                                      scale=np.sqrt(2/(n_input)))
        if self.b is None:
            self.b = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_W = self.optimizer
        self.optimizer_b = copy_class(self.optimizer)
    
    def Forward(self, a):
        self.a = a
        return self.W @ a + self.b

    def Backward(self, delta_Z):
        dg_W = delta_Z @ self.a.T
        dg_b = np.sum(delta_Z, axis=-1)
        last_delta = self.W.T @ delta_Z
        self.W += self.optimizer_W.Delta(dg_W)
        self.b += self.optimizer_b.Delta(dg_b).reshape(-1, 1)
        return last_delta