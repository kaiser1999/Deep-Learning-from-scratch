import numpy as np
import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))
from .Intermediate_Layers import Layer

class RNN_Activation:
    def __init__(self, a_func):
        self.a_func = a_func.lower()
        self.const = 709.78     # max exp of np.float64 value
        assert self.a_func in ["relu", "sigmoid", "tanh", "linear"]
    
    def get_act(self, z):
        if self.a_func == "relu":
            return z * (z > 0)
        elif self.a_func == "sigmoid":
            return 1. / (1. + self.exp_clip(-z))
        elif self.a_func == "tanh":
            exp_z = self.exp_clip(z)
            exp_neg_z = self.exp_clip(-z)
            return (exp_z - exp_neg_z)/(exp_z + exp_neg_z)
        elif self.a_func == "linear":
            return z
    
    def exp_clip(self, x):
        return np.exp(np.clip(x, a_min=None, a_max=self.const))
    
    def get_deri(self, A, Z):
        if self.a_func == "relu":
            return A * (Z > 0)
        elif self.a_func == "sigmoid":
            return A * (1. - A)
        elif self.a_func == "tanh":
            return 1. - A**2
        elif self.a_func == "linear":
            return 1.

def W_Initializer(d_ell, n_feature):
    return np.random.randn(d_ell, n_feature) * np.sqrt(2 / (d_ell + n_feature))

def V_Initializer(d_ell):
    return np.random.randn(d_ell, d_ell) * np.sqrt(2 / (d_ell + d_ell))

#%%
class SimpleRNN(Layer):
    def __init__(self, d_ell, a_func="tanh", return_sequence=False, graph=True):
        self.d_ell = d_ell
        self.W_h, self.V_h, self.b_h, self.h_0 = None, None, None, None
        self.f_func = RNN_Activation(a_func.lower())
        self.return_seq = return_sequence
        if graph or not self.return_seq:
            self.Backward_func = self.Backward_GRAPH
        else:
            self.Backward_func = self.Backward_BOOK_many2many
    
    def Param_init(self, input_shape):
        self.seq_len, n_feature, self.bs = input_shape
        if self.W_h is None: self.W_h = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_h = self.optimizer
        
        if self.V_h is None: self.V_h = V_Initializer(self.d_ell)
        self.optimizer_V_h = copy_class(self.optimizer)
        
        if self.b_h is None: self.b_h = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_h = copy_class(self.optimizer)
        
        if self.h_0 is None: self.h_0 = np.random.randn(self.d_ell) * np.sqrt(2 / self.d_ell)
        self.optimizer_h_0 = copy_class(self.optimizer)
    
    def Forward(self, x):
        self.x = x
        self.M = np.shape(x)[-1]
        self.seq_len, self.n_feature, self.bs = np.shape(x)
        if self.h_0 is None:
            self.Param_init(self.seq_len, self.n_feature)
        
        self.h = np.zeros((self.seq_len+1, self.d_ell, self.bs))
        self.z_h = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.h[0,:] = np.repeat(self.h_0, self.bs).reshape(-1, self.bs)
        for t in range(self.seq_len):
            self.z_h[t,:] = self.W_h @ x[t,:] + self.V_h @ self.h[t,:] + self.b_h
            self.h[t+1,:] = self.f_func.get_act(self.z_h[t,:])
        
        if self.return_seq:
            return self.h[1:,:]     # Return entire sequence
        else:
            return self.h[-1,:]     # Return only last timestep
    
    def Backward(self, delta_h):
        dg_W_h, dg_V_h, dg_b_h, dg_h_0, delta_x = self.Backward_func(delta_h)
        
        self.W_h += self.optimizer_W_h.Delta(dg_W_h)
        self.V_h += self.optimizer_V_h.Delta(dg_V_h)
        self.b_h += self.optimizer_b_h.Delta(dg_b_h)
        self.h_0 += self.optimizer_h_0.Delta(dg_h_0)
        return delta_x
        
    def Backward_GRAPH(self, delta_h):
        err, dg_W_h, dg_V_h, dg_b_h = 0, 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        dg_h = self.f_func.get_deri(self.h[1:,:], self.z_h)
        
        if not self.return_seq:
            err = delta_h
            
        for t in reversed(range(self.seq_len)):
            if self.return_seq:
                err = dg_h[t,:] * (delta_h[t,:] + err)
            else:
                err = dg_h[t,:] * err
            
            delta_x[t,:] = self.W_h.T @ err
            dg_W_h += err @ self.x[t,:].T
            dg_V_h += err @ self.h[t,:].T
            dg_b_h += np.sum(err, axis=-1).reshape(-1, 1)
            
            err = self.V_h.T @ err
        
        dg_h_0 = np.sum(err, axis=-1)
        return dg_W_h, dg_V_h, dg_b_h, dg_h_0, delta_x
    
    def Backward_BOOK_many2many(self, delta_h):
        dg_W_h, dg_V_h, dg_b_h = 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        dg_h = self.f_func.get_deri(self.h[1:,:], self.z_h)
        
        for t in reversed(range(self.seq_len)):
            delta_tau = delta_h[t,:]
            for tau in reversed(range(t+1)):
                delta_tau = dg_h[tau,:] * delta_tau
                # Dont use delta_tau *= dg_h[tau,:], it changes delta_h
                
                dg_W_h += delta_tau @ self.x[tau,:].T
                dg_V_h += delta_tau @ self.h[tau,:].T
                dg_b_h += np.sum(delta_tau, axis=-1).reshape(-1, 1)
                delta_tau = self.V_h.T @ delta_tau

            err = 0
            for tau in reversed(range(t+1, self.seq_len)):
                err = dg_h[tau,:] * (delta_h[tau,:] + err)
                err = self.V_h.T @ err
            
            err = dg_h[t,:] * (delta_h[t,:] + err)
            delta_x[t,:] = self.W_h.T @ err
        
        dg_h_0 = np.sum(self.V_h.T @ err, axis=-1)
        return dg_W_h, dg_V_h, dg_b_h, dg_h_0, delta_x

#%%
class GRU(Layer):
    def __init__(self, d_ell, a_func="tanh", recurrent_func="sigmoid", return_sequence=False, graph=True):
        self.d_ell = d_ell
        self.W_r, self.V_r, self.b_r = None, None, None
        self.W_u, self.V_u, self.b_u = None, None, None
        self.W_tilde_h, self.V_tilde_h, self.b_tilde_h, self.h_0 = None, None, None, None

        self.g_func = RNN_Activation(a_func.lower())
        self.sig_func = copy_class(RNN_Activation(recurrent_func.lower()))
        self.return_seq = return_sequence
        if graph or not self.return_seq:
            self.Backward_func = self.Backward_GRAPH
        else:
            self.Backward_func = self.Backward_BOOK_many2many
    
    def Param_init(self, input_shape):
        self.seq_len, n_feature, self.bs = input_shape
        if self.W_r is None: self.W_r = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_r = copy_class(self.optimizer)
        
        if self.W_u is None: self.W_u = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_u = copy_class(self.optimizer)
        
        if self.W_tilde_h is None: self.W_tilde_h = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_tilde_h = copy_class(self.optimizer)
        
        if self.V_r is None: self.V_r = V_Initializer(self.d_ell)
        self.optimizer_V_r = copy_class(self.optimizer)
        
        if self.V_u is None: self.V_u = V_Initializer(self.d_ell)
        self.optimizer_V_u = copy_class(self.optimizer)
        
        if self.V_tilde_h is None: self.V_tilde_h = V_Initializer(self.d_ell)
        self.optimizer_V_tilde_h = copy_class(self.optimizer)
        
        if self.b_r is None: self.b_r = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_r = copy_class(self.optimizer)
        
        if self.b_u is None: self.b_u = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_u = copy_class(self.optimizer)
        
        if self.b_tilde_h is None: self.b_tilde_h = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_tilde_h = copy_class(self.optimizer)

        if self.h_0 is None: self.h_0 = np.random.randn(self.d_ell) * np.sqrt(2 / self.d_ell)
        self.optimizer_h_0 = copy_class(self.optimizer)
    
    def Forward(self, x):
        self.x = x
        self.M = np.shape(x)[-1]  
        self.z_r = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.z_u = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.z_tilde_h = np.zeros((self.seq_len, self.d_ell, self.bs))
        
        self.r = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.u = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.tilde_h = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.h = np.zeros((self.seq_len+1, self.d_ell, self.bs))
        
        self.h[0,:] = np.repeat(self.h_0, self.bs).reshape(-1, self.bs)
        for t in range(self.seq_len):
            self.z_r[t,:] = self.W_r @ x[t,:] + self.V_r @ self.h[t,:] + self.b_r
            self.r[t,:] = self.sig_func.get_act(self.z_r[t,:])
            
            self.z_u[t,:] = self.W_u @ x[t,:] + self.V_u @ self.h[t,:] + self.b_u
            self.u[t,:] = self.sig_func.get_act(self.z_u[t,:])
            
            self.z_tilde_h[t,:] = self.W_tilde_h @ x[t,:] + self.V_tilde_h @ (self.r[t,:] * self.h[t,:]) + self.b_tilde_h
            self.tilde_h[t,:] = self.g_func.get_act(self.z_tilde_h[t,:])
            
            self.h[t+1,:] = (1 - self.u[t,:]) * self.h[t,:] + self.u[t,:] * self.tilde_h[t,:]
        
        if self.return_seq:
            return self.h[1:,:]     # Return entire sequence
        else:
            return self.h[-1,:]     # Return only last timestep
    
    def Backward(self, delta_h):
        (dg_W_r, dg_V_r, dg_b_r, dg_W_u, dg_V_u, dg_b_u, 
         dg_W_tilde_h, dg_V_tilde_h, dg_b_tilde_h, dg_h_0, delta_x) = self.Backward_func(delta_h)
        
        self.W_r += self.optimizer_W_r.Delta(dg_W_r)
        self.V_r += self.optimizer_V_r.Delta(dg_V_r)
        self.b_r += self.optimizer_b_r.Delta(dg_b_r)
        self.W_u += self.optimizer_W_u.Delta(dg_W_u)
        self.V_u += self.optimizer_V_u.Delta(dg_V_u)
        self.b_u += self.optimizer_b_u.Delta(dg_b_u)
        self.W_tilde_h += self.optimizer_W_tilde_h.Delta(dg_W_tilde_h)
        self.V_tilde_h += self.optimizer_V_tilde_h.Delta(dg_V_tilde_h)
        self.b_tilde_h += self.optimizer_b_tilde_h.Delta(dg_b_tilde_h)
        self.h_0 += self.optimizer_h_0.Delta(dg_h_0)
        
        return delta_x

    def Backward_GRAPH(self, delta_h):
        dg_W_r, dg_V_r, dg_b_r = 0, 0, 0
        dg_W_u, dg_V_u, dg_b_u = 0, 0, 0
        dg_W_tilde_h, dg_V_tilde_h, dg_b_tilde_h = 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        
        delta_r = self.sig_func.get_deri(self.r, self.z_r)
        delta_u = self.sig_func.get_deri(self.u, self.z_u)
        delta_tilde_h = self.g_func.get_deri(self.tilde_h, self.z_tilde_h)
        
        err = 0
        if not self.return_seq:
            err = delta_h
            
        for t in reversed(range(self.seq_len)):
            if self.return_seq:
                err += delta_h[t,:]
            
            err_tilde_h = err * delta_tilde_h[t,:] * self.u[t,:]
            
            err_r = delta_r[t,:] * (self.V_tilde_h.T @ err_tilde_h)
            dg_W_r += err_r @ self.x[t,:].T
            dg_V_r += err_r @ self.h[t,:].T
            dg_b_r += np.sum(err_r, axis=-1).reshape(-1, 1)
            
            err_u = delta_u[t,:] * err
            err_u *= self.tilde_h[t,:] - self.h[t,:]
            dg_W_u += err_u @ self.x[t,:].T
            dg_V_u += err_u @ self.h[t,:].T
            dg_b_u += np.sum(err_u, axis=-1).reshape(-1, 1)
            
            dg_W_tilde_h += err_tilde_h @ self.x[t,:].T
            dg_V_tilde_h += err_tilde_h @ self.h[t,:].T
            dg_b_tilde_h += np.sum(err_tilde_h, axis=-1).reshape(-1, 1)
            
            Term_W_r = self.V_tilde_h.T @ err_tilde_h
            Term_W_r *= delta_r[t,:] * self.h[t,:]
            Term_W_r = self.W_r.T @ Term_W_r
            delta_x[t,:] = self.W_u.T @ err_u + Term_W_r + self.W_tilde_h.T @ err_tilde_h
            
            Term_V_r = self.V_tilde_h.T @ err_tilde_h
            err = (self.V_u.T @ err_u + err * (1 - self.u[t,:]) + 
                   self.V_r.T @ (delta_r[t,:] * self.h[t,:] * Term_V_r) + 
                   self.r[t,:] * Term_V_r)
        
        dg_h_0 = np.sum(err, axis=-1)
        return (dg_W_r, dg_V_r, dg_b_r, dg_W_u, dg_V_u, dg_b_u, dg_W_tilde_h, 
                dg_V_tilde_h, dg_b_tilde_h, dg_h_0, delta_x)
    
    def Backward_BOOK_many2many(self, delta_h):
        dg_W_r, dg_V_r, dg_b_r = 0, 0, 0
        dg_W_u, dg_V_u, dg_b_u = 0, 0, 0
        dg_W_tilde_h, dg_V_tilde_h, dg_b_tilde_h = 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        
        delta_r = self.sig_func.get_deri(self.r, self.z_r)
        delta_u = self.sig_func.get_deri(self.u, self.z_u)
        delta_tilde_h = self.g_func.get_deri(self.tilde_h, self.z_tilde_h)
        for t in reversed(range(self.seq_len)):
            delta_tau = delta_h[t,:]
            for tau in reversed(range(t+1)):
                err_tilde_h = delta_tilde_h[tau,:] * delta_tau * self.u[tau,:]
                
                err_r = delta_r[tau,:] * (self.V_tilde_h.T @ err_tilde_h)
                dg_W_r += err_r @ self.x[tau,:].T
                dg_V_r += err_r @ self.h[tau,:].T
                
                dg_b_r += np.sum(err_r, axis=-1).reshape(-1, 1)
                
                err_u = delta_u[tau,:] * delta_tau
                err_u *= self.tilde_h[tau,:] - self.h[tau,:]
                dg_W_u += err_u @ self.x[tau,:].T
                dg_V_u += err_u @ self.h[tau,:].T
                dg_b_u += np.sum(err_u, axis=-1).reshape(-1, 1)
                
                dg_W_tilde_h += err_tilde_h @ self.x[tau,:].T
                dg_V_tilde_h += err_tilde_h @ self.h[tau,:].T
                dg_b_tilde_h += np.sum(err_tilde_h, axis=-1).reshape(-1, 1)
            
                Term_V_r = self.V_tilde_h.T @ err_tilde_h
                delta_tau = (self.V_u.T @ err_u + delta_tau * (1 - self.u[tau,:]) + 
                             self.V_r.T @ (delta_r[tau,:] * self.h[tau,:] * Term_V_r) + 
                             self.r[tau,:] * Term_V_r)
            
            err = 0
            for tau in reversed(range(t, self.seq_len)):
                err += delta_h[tau,:]
                err_tilde_h = delta_tilde_h[tau,:] * err * self.u[tau,:]
                err_u = delta_u[tau,:] * err
                err_u *= self.tilde_h[tau,:] - self.h[tau,:]
                
                Term_V_r = self.V_tilde_h.T @ err_tilde_h
                err = (self.V_u.T @ err_u + err * (1 - self.u[tau,:]) + 
                       self.V_r.T @ (delta_r[tau,:] * self.h[tau,:] * Term_V_r) + 
                       self.r[tau,:] * Term_V_r)
            
            Term_W_r = self.V_tilde_h.T @ err_tilde_h
            Term_W_r *= delta_r[t,:] * self.h[t,:]
            Term_W_r = self.W_r.T @ Term_W_r
            delta_x[t,:] = self.W_u.T @ err_u + Term_W_r + self.W_tilde_h.T @ err_tilde_h
        
        dg_h_0 = np.sum(err, axis=-1)
        return (dg_W_r, dg_V_r, dg_b_r, dg_W_u, dg_V_u, dg_b_u, 
                dg_W_tilde_h, dg_V_tilde_h, dg_b_tilde_h, dg_h_0, delta_x)

#%%    
class LSTM(Layer):
    def __init__(self, d_ell, a_func="tanh", recurrent_func="sigmoid", return_sequence=False, graph=True):
        self.d_ell = d_ell
        self.W_f, self.V_f, self.b_f = None, None, None
        self.W_i, self.V_i, self.b_i = None, None, None
        self.W_tilde_c, self.V_tilde_c, self.b_tilde_c = None, None, None
        self.W_o, self.V_o, self.b_o, self.h_0, self.c_0 = None, None, None, None, None

        self.g_func = RNN_Activation(a_func.lower())
        self.sig_func = copy_class(RNN_Activation(recurrent_func.lower()))
        self.return_seq = return_sequence
        if graph or not self.return_seq:
            self.Backward_func = self.Backward_GRAPH
        else:
            self.Backward_func = self.Backward_BOOK_many2many

    def Param_init(self, input_shape):
        self.seq_len, n_feature, self.bs = input_shape
        if self.W_f is None: self.W_f = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_f = copy_class(self.optimizer)
        
        if self.W_i is None: self.W_i = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_i = copy_class(self.optimizer)
        
        if self.W_tilde_c is None: self.W_tilde_c = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_tilde_c = copy_class(self.optimizer)
        
        if self.W_o is None: self.W_o = W_Initializer(self.d_ell, n_feature)
        self.optimizer_W_o = copy_class(self.optimizer)
        
        if self.V_f is None: self.V_f = V_Initializer(self.d_ell)
        self.optimizer_V_f = copy_class(self.optimizer)
        
        if self.V_i is None: self.V_i = V_Initializer(self.d_ell)
        self.optimizer_V_i = copy_class(self.optimizer)
        
        if self.V_tilde_c is None: self.V_tilde_c = V_Initializer(self.d_ell)
        self.optimizer_V_tilde_c = copy_class(self.optimizer)
        
        if self.V_o is None: self.V_o = V_Initializer(self.d_ell)
        self.optimizer_V_o = copy_class(self.optimizer)
        
        if self.b_f is None: self.b_f = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_f = copy_class(self.optimizer)
        
        if self.b_i is None: self.b_i = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_i = copy_class(self.optimizer)
        
        if self.b_tilde_c is None: self.b_tilde_c = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_tilde_c = copy_class(self.optimizer)
        
        if self.b_o is None: self.b_o = np.zeros(self.d_ell).reshape(-1, 1)
        self.optimizer_b_o = copy_class(self.optimizer)

        if self.c_0 is None: self.c_0 = np.random.randn(self.d_ell) * np.sqrt(2 / self.d_ell)
        self.optimizer_c_0 = copy_class(self.optimizer)
        
        if self.h_0 is None: self.h_0 = np.random.randn(self.d_ell) * np.sqrt(2 / self.d_ell)
        self.optimizer_h_0 = copy_class(self.optimizer)
    
    def Forward(self, x):
        self.x = x
        self.M = np.shape(x)[-1]
        self.z_f = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.z_i = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.z_tilde_c = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.z_o = np.zeros((self.seq_len, self.d_ell, self.bs))
        
        self.f = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.i = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.tilde_c = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.o = np.zeros((self.seq_len, self.d_ell, self.bs))
        self.c = np.zeros((self.seq_len+1, self.d_ell, self.bs))
        self.h = np.zeros((self.seq_len+1, self.d_ell, self.bs))
        self.g_c = np.zeros((self.seq_len, self.d_ell, self.bs))
        
        self.c[0,:] = np.repeat(self.c_0, self.bs).reshape(-1, self.bs)
        self.h[0,:] = np.repeat(self.h_0, self.bs).reshape(-1, self.bs)
        for t in range(self.seq_len):
            self.z_f[t,:] = self.W_f @ x[t,:] + self.V_f @ self.h[t,:] + self.b_f
            self.f[t,:] = self.sig_func.get_act(self.z_f[t,:])
            
            self.z_i[t,:] = self.W_i @ x[t,:] + self.V_i @ self.h[t,:] + self.b_i
            self.i[t,:] = self.sig_func.get_act(self.z_i[t,:])
            
            self.z_tilde_c[t,:] = self.W_tilde_c @ x[t,:] + self.V_tilde_c @ self.h[t,:] + self.b_tilde_c
            self.tilde_c[t,:] = self.g_func.get_act(self.z_tilde_c[t,:])
            
            self.z_o[t,:] = self.W_o @ x[t,:] + self.V_o @ self.h[t,:] + self.b_o
            self.o[t,:] = self.sig_func.get_act(self.z_o[t,:])
            
            self.c[t+1,:] = self.i[t,:] * self.tilde_c[t,:] + self.f[t,:] * self.c[t,:]
            self.g_c[t,:] = self.g_func.get_act(self.c[t+1,:])
            self.h[t+1,:] = self.o[t,:] * self.g_c[t,:]
        
        if self.return_seq:
            return self.h[1:,:]     # Return entire sequence
        else:
            return self.h[-1,:]     # Return only last timestep
        
    def Backward(self, delta_h):
        (dg_W_f, dg_V_f, dg_b_f, dg_W_i, dg_V_i, dg_b_i, 
         dg_W_tilde_c, dg_V_tilde_c, dg_b_tilde_c, dg_W_o, dg_V_o, dg_b_o,
         dg_h_0, dg_c_0, delta_x) = self.Backward_func(delta_h)
        
        '''
        if self.return_seq:
            (a, b, c, d, e, f, 
             g, h, i, j, k, l,
             m, n, o) = self.Backward_BOOK_many2many(delta_h)
            
            print(np.sum((a - dg_W_f)**2))
            print(np.sum((j - dg_W_o)**2))
            print(np.sum((n - dg_c_0)**2))
            print(np.sum((o - delta_x)**2))
            '''
        
        self.W_f += self.optimizer_W_f.Delta(dg_W_f)
        self.V_f += self.optimizer_V_f.Delta(dg_V_f)
        self.b_f += self.optimizer_b_f.Delta(dg_b_f)
        self.W_i += self.optimizer_W_i.Delta(dg_W_i)
        self.V_i += self.optimizer_V_i.Delta(dg_V_i)
        self.b_i += self.optimizer_b_i.Delta(dg_b_i)
        self.W_tilde_c += self.optimizer_W_tilde_c.Delta(dg_W_tilde_c)
        self.V_tilde_c += self.optimizer_V_tilde_c.Delta(dg_V_tilde_c)
        self.b_tilde_c += self.optimizer_b_tilde_c.Delta(dg_b_tilde_c)
        self.W_o += self.optimizer_W_o.Delta(dg_W_o)
        self.V_o += self.optimizer_V_o.Delta(dg_V_o)
        self.b_o += self.optimizer_b_o.Delta(dg_b_o)
        self.c_0 += self.optimizer_c_0.Delta(dg_c_0)
        self.h_0 += self.optimizer_h_0.Delta(dg_h_0)
        
        return delta_x
    
    def Backward_GRAPH(self, delta_h):
        dg_W_f, dg_V_f, dg_b_f = 0, 0, 0
        dg_W_i, dg_V_i, dg_b_i = 0, 0, 0
        dg_W_tilde_c, dg_V_tilde_c, dg_b_tilde_c = 0, 0, 0
        dg_W_o, dg_V_o, dg_b_o = 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        
        delta_f = self.sig_func.get_deri(self.f, self.z_f)
        delta_i = self.sig_func.get_deri(self.i, self.z_i)
        delta_tilde_c = self.g_func.get_deri(self.tilde_c, self.z_tilde_c)
        delta_o = self.sig_func.get_deri(self.o, self.z_o)
        delta_c = self.g_func.get_deri(self.g_c, self.c[1:])
        
        err, lamb = 0, 0
        if not self.return_seq:
            err = delta_h
        for t in reversed(range(self.seq_len)):
            if self.return_seq:
                err += delta_h[t,:]
            
            Term = delta_c[t,:] * self.o[t,:]
            err_term = lamb + Term * err
            
            err_f = self.c[t,:] * delta_f[t,:] * err_term
            dg_W_f += err_f @ self.x[t,:].T
            dg_V_f += err_f @ self.h[t,:].T
            dg_b_f += np.sum(err_f, axis=-1).reshape(-1, 1)
            
            err_i = self.tilde_c[t,:] * delta_i[t,:] * err_term
            dg_W_i += err_i @ self.x[t,:].T
            dg_V_i += err_i @ self.h[t,:].T
            dg_b_i += np.sum(err_i, axis=-1).reshape(-1, 1)
            
            err_tilde_c =  self.i[t,:] * delta_tilde_c[t,:] * err_term
            dg_W_tilde_c += err_tilde_c @ self.x[t,:].T
            dg_V_tilde_c += err_tilde_c @ self.h[t,:].T
            dg_b_tilde_c += np.sum(err_tilde_c, axis=-1).reshape(-1, 1)
            
            err_o = self.g_c[t,:] * delta_o[t,:] * err
            dg_W_o += err_o @ self.x[t,:].T
            dg_V_o += err_o @ self.h[t,:].T
            dg_b_o += np.sum(err_o, axis=-1).reshape(-1, 1)
            
            Term_W_i = self.W_i.T @ err_i
            Term_W_f = self.W_f.T @ err_f
            Term_W_tilde_c = self.W_tilde_c.T @ err_tilde_c
            Term_W_o = self.W_o.T @ err_o
            delta_x[t,:] = Term_W_o + Term_W_i + Term_W_f + Term_W_tilde_c
            
            lamb = self.f[t,:] * err_term
            
            Term_V_i = self.V_i.T @ err_i
            Term_V_f = self.V_f.T @ err_f
            Term_V_tilde_c = self.V_tilde_c.T @ err_tilde_c
            Term_V_o = self.V_o.T @ err_o
            err = Term_V_o + Term_V_i + Term_V_f + Term_V_tilde_c
            
        dg_h_0 = np.sum(err, axis=-1)
        dg_c_0 = np.sum(lamb, axis=-1)
        return (dg_W_f, dg_V_f, dg_b_f, dg_W_i, dg_V_i, dg_b_i, 
                dg_W_tilde_c, dg_V_tilde_c, dg_b_tilde_c, dg_W_o, dg_V_o, dg_b_o,
                dg_h_0, dg_c_0, delta_x)
    
    def Backward_BOOK_many2many(self, delta_h):
        dg_W_f, dg_V_f, dg_b_f = 0, 0, 0
        dg_W_i, dg_V_i, dg_b_i = 0, 0, 0
        dg_W_tilde_c, dg_V_tilde_c, dg_b_tilde_c = 0, 0, 0
        dg_W_o, dg_V_o, dg_b_o = 0, 0, 0
        delta_x = np.zeros(np.shape(self.x))
        
        delta_f = self.sig_func.get_deri(self.f, self.z_f)
        delta_i = self.sig_func.get_deri(self.i, self.z_i)
        delta_tilde_c = self.g_func.get_deri(self.tilde_c, self.z_tilde_c)
        delta_o = self.sig_func.get_deri(self.o, self.z_o)
        delta_c = self.g_func.get_deri(self.g_c, self.c[1:])
        
        for t in reversed(range(self.seq_len)):
            delta_tau = delta_h[t,:].copy()
            lamb_tau = 0
            for tau in reversed(range(t+1)):
                Term = delta_c[tau,:] * self.o[tau,:]
                delta_term = lamb_tau + Term * delta_tau
                
                err_f = self.c[tau,:] * delta_f[tau,:] * delta_term
                dg_W_f += err_f @ self.x[tau,:].T
                dg_V_f += err_f @ self.h[tau,:].T
                dg_b_f += np.sum(err_f, axis=-1).reshape(-1, 1)
                
                err_i = self.tilde_c[tau,:] * delta_i[tau,:] * delta_term
                dg_W_i += err_i @ self.x[tau,:].T
                dg_V_i += err_i @ self.h[tau,:].T
                dg_b_i += np.sum(err_i, axis=-1).reshape(-1, 1)
                
                err_tilde_c = self.i[tau,:] * delta_tilde_c[tau,:] * delta_term
                dg_W_tilde_c += err_tilde_c @ self.x[tau,:].T
                dg_V_tilde_c += err_tilde_c @ self.h[tau,:].T
                dg_b_tilde_c += np.sum(err_tilde_c, axis=-1).reshape(-1, 1)
                
                err_o = self.g_c[tau,:] * delta_o[tau,:] * delta_tau
                dg_W_o += err_o @ self.x[tau,:].T
                dg_V_o += err_o @ self.h[tau,:].T
                dg_b_o += np.sum(err_o, axis=-1).reshape(-1, 1)
                
                lamb_tau = self.f[tau,:] * delta_term
                
                Term_V_i = self.V_i.T @ err_i
                Term_V_f = self.V_f.T @ err_f
                Term_V_tilde_c = self.V_tilde_c.T @ err_tilde_c
                Term_V_o = self.V_o.T @ err_o
                delta_tau = Term_V_o + Term_V_i + Term_V_f + Term_V_tilde_c
                
            err, lamb = 0, 0
            for tau in reversed(range(t, self.seq_len)):
                err += delta_h[tau,:]
                
                Term = delta_c[tau,:] * self.o[tau,:]
                err_term = lamb + Term * err
                
                err_i = self.tilde_c[tau,:] * delta_i[tau,:] * err_term
                err_f = self.c[tau,:] * delta_f[tau,:] * err_term
                err_tilde_c = self.i[tau,:] * delta_tilde_c[tau,:] * err_term
                err_o = self.g_c[tau,:] * delta_o[tau,:] * err
                
                lamb = self.f[tau,:] * err_term
                
                Term_V_i = self.V_i.T @ err_i
                Term_V_f = self.V_f.T @ err_f
                Term_V_tilde_c = self.V_tilde_c.T @ err_tilde_c
                Term_V_o = self.V_o.T @ err_o
                err = Term_V_o + Term_V_i + Term_V_f + Term_V_tilde_c
            
            Term_W_o = self.W_o.T @ err_o
            Term_W_i = self.W_i.T @ err_i
            Term_W_f = self.W_f.T @ err_f 
            Term_W_tilde_c = self.W_tilde_c.T @ err_tilde_c
            delta_x[t,:] = Term_W_o + Term_W_i + Term_W_f + Term_W_tilde_c
            
        dg_h_0 = np.sum(err, axis=-1)
        dg_c_0 = np.sum(lamb, axis=-1)
        return (dg_W_f, dg_V_f, dg_b_f, dg_W_i, dg_V_i, dg_b_i, 
                dg_W_tilde_c, dg_V_tilde_c, dg_b_tilde_c, dg_W_o, dg_V_o, dg_b_o,
                dg_h_0, dg_c_0, delta_x)
    

#%%
from Enhanced_Gradient_Descent import Adam

if __name__ == "__main__":
    np.random.seed(4012)
    seq = True
    model = "lstm"
    x = np.arange(12).reshape(4, 3)
    x = np.repeat(x, 2).reshape(4, 3, 2)
    
    if seq:
        err = np.arange(20).reshape(4, 5)
        err = np.repeat(err, 2).reshape(4, 5, 2)
    else:
        err = np.arange(5).reshape(1, 5)
        err = np.repeat(err, 2).reshape(5, 2)
    
    np.random.seed(4012)
    if model.lower() == "rnn":
        A = SimpleRNN(5, return_sequence=seq, graph=True)
    elif model.lower() == "gru":
        A = GRU(5, return_sequence=seq, graph=True)
    else:
        A = LSTM(5, return_sequence=seq, graph=True)
    A.optimizer = Adam(1e-3)
    
    A.Forward(x)
    print(A.Backward(err))
    
    np.random.seed(4012)
    if model.lower() == "rnn":
        B = SimpleRNN(5, return_sequence=seq, graph=False)
    elif model.lower() == "gru":
        B = GRU(5, return_sequence=seq, graph=False)
    else:
        B = LSTM(5, return_sequence=seq, graph=False)
    B.optimizer = Adam(1e-3)
    
    B.Forward(x)
    print(B.Backward(err))
    