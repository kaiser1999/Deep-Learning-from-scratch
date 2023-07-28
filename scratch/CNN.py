import numpy as np
import pickle
from skimage.filters import correlate_sparse as correlate # valid mode not avaiable in ndimage
from .Intermediate_Layers import Layer
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))

class Flattening(Layer):
    def Forward(self, X):
        self.shape = np.shape(X)
        return X.reshape(-1, np.shape(X)[-1])
    
    def Backward(self, delta_F):
        return delta_F.reshape(self.shape)
    
class GlobalAveragePooling():
    def Forward(self, X):
        self.shape = np.shape(X)
        return np.average(X, axis=(0, 1))
    
    def Backward(self, delta_F):
        K = np.prod(self.shape[:2])
        delta_X = np.repeat(delta_F / K, K)
        return delta_X.reshape(self.shape[::-1]).T
    
class Convolutional(Layer):
    # X: s_i1 x s_i2 x n_v x M
    # K: s_f1 x s_f2 x n_v x n_f
    # Z: s_o1 x s_o2 x n_f x M
    def __init__(self, n_f, s_f=[3, 3], stride=[1, 1]):
        self.n_f = n_f
        self.s_f = np.array(s_f)
        self.stride = np.array(stride)
        self.b, self.K = None, None
        assert self.n_f > 0 and isinstance(self.n_f, int)
        assert len(s_f) == 2 and len(stride) == 2
        assert np.all(self.s_f > 0) and np.all([isinstance(val, int) for val in s_f])
        assert np.all(self.stride > 0) and np.all([isinstance(val, int) for val in stride])
    
    def _get_pad_size(self, s_i, s_f, s_sr, s_o=None):
        s_i, s_f, s_sr = np.array(s_i), np.array(s_f), np.array(s_sr)
        if s_o is None:
            s_o = np.ceil(s_i/s_sr)
        else:
            s_o = np.array(s_o)
            
        Total_PAD = np.clip((s_o - 1) * s_sr + s_f - s_i, a_min=0, a_max=None).astype(int)
        PAD_Neg = Total_PAD // 2
        PAD_Pos = Total_PAD - PAD_Neg
        return PAD_Neg, PAD_Pos
    
    def Param_init(self, input_shape):
        *s_i, self.n_v, self.bs = input_shape
        n_input = tuple(self.s_f) + (self.n_v, self.n_f)
        if self.K is None:
            self.K = np.random.randn(*n_input) * np.sqrt(2 / np.sum(n_input))
        if self.b is None:
            self.b = np.zeros(self.n_f).reshape(-1, 1)
        self.optimizer_K = self.optimizer
        self.optimizer_b = copy_class(self.optimizer)
        
        PAD_Neg, PAD_Pos = self._get_pad_size(s_i, self.s_f, self.stride)
        self.PAD_Forw_shape = tuple(zip(PAD_Neg, PAD_Pos)) + ((0, 0), (0, 0))
        
        remainder = np.mod(s_i + PAD_Neg + PAD_Pos - self.s_f, self.stride)
        s_o = np.floor((s_i + PAD_Neg + PAD_Pos - self.s_f)/self.stride) + 1
        s_Z = tuple((s_o.astype(int) - 1)*self.stride + 1 + remainder)
        self.delta_X_shape = tuple(s_i) + (self.n_v,)
        self.bar_delta_Z_shape = s_Z + (self.n_f,)
        
        PAD_Back_Neg, PAD_Back_Pos = self._get_pad_size(s_Z, self.s_f, (1, 1), s_i)
        # Reversed order in Backpropagation
        self.PAD_Back_shape = tuple(zip(PAD_Back_Pos, PAD_Back_Neg)) + ((0, 0), (0, 0))
    
    def Forward(self, X):
        self.tilde_X = np.pad(X, self.PAD_Forw_shape, "constant")
        self.M = np.shape(X)[-1]
        Z, rot_K = [], np.rot90(self.K, 2, (1, 0))
        for r in range(self.n_f):
            conv_X_K = self.b[r][0]
            for h in range(self.n_v):
                # Convolution = Cross-correlation + Rotated Kernel
                expand_rot_K = np.expand_dims(rot_K[:,:,h,r], axis=-1)
                conv_X_K += correlate(self.tilde_X[:,:,h,:], expand_rot_K, mode="valid")
            Z.append(conv_X_K)
        
        return np.moveaxis(Z, 0, 2)[::self.stride[0], ::self.stride[1], :, :]

    def Backward(self, delta_Z):
        delta_X = np.zeros((self.delta_X_shape + (self.M,)))
        bar_delta_Z = np.zeros((self.bar_delta_Z_shape + (self.M,)))
        bar_delta_Z[::self.stride[0], ::self.stride[1], :, :] = delta_Z
        tilde_bar_delta_Z = np.pad(bar_delta_Z, self.PAD_Back_shape, mode="constant")
        
        for h in range(self.n_v):
            for r in range(self.n_f):
                expand_K = np.expand_dims(self.K[:,:,h,r], axis=-1)
                delta_X[:,:,h,:] += correlate(tilde_bar_delta_Z[:,:,r,:], expand_K, mode="valid")
            
        dg_K = np.zeros(np.shape(self.K))
        for r in range(self.n_f):
            for h in range(self.n_v):
                corr_X_dZ = correlate(self.tilde_X[:,:,h,:], bar_delta_Z[:,:,r,:], mode="valid")
                dg_K[:,:,h,r] = np.squeeze(corr_X_dZ, axis=-1)
            
        dg_K = np.rot90(dg_K, 2, axes=(1,0))
        dg_b = np.sum(delta_Z, axis=(0,1,3))
        self.K += self.optimizer_K.Delta(dg_K)
        self.b += self.optimizer_b.Delta(dg_b).reshape(-1,1)
        return delta_X

class Pooling(Layer):
    # X: s_i1 x s_i2 x n_f x M
    def __init__(self, p_func, s_q=[2, 2]):
        self.p_func = p_func.lower()
        self.s_q = np.array(s_q)
        self.Shape_X = (self.s_q[0], self.s_q[1], 1, 1)
        self.max_mat, self.Shape_P = None, None
        assert self.p_func in ["maxpooling", "averagepooling"]
        assert np.all(self.s_q > 0) and np.all([isinstance(val, int) for val in s_q])
    
    def Param_init(self, input_shape):
        *s_i, n_v, bs = input_shape
        
        mat_val = np.arange(np.prod(self.s_q))[::-1].reshape(self.Shape_X) + 1 
        ceil_1, ceil_2 = np.ceil(s_i/self.s_q).astype(int)
        max_ones = np.ones((ceil_1, ceil_2, 1, 1))
        self.max_mat = np.kron(max_ones, mat_val)
        self.Shape_P = (ceil_1, self.s_q[0], ceil_2, self.s_q[1], n_v, -1)
        
        Total_PAD = np.ceil(s_i/self.s_q).astype(int)*self.s_q - s_i
        PAD_Neg = Total_PAD//2
        PAD_Pos = Total_PAD - PAD_Neg
        self.PAD_shape = tuple(zip(PAD_Neg, PAD_Pos)) + ((0, 0), (0, 0))
        self.k_Left, self.k_Top = PAD_Neg
        self.s_Right, self.s_Bottom = s_i + PAD_Neg
    
    def Forward(self, X):
        self.X = np.pad(X, self.PAD_shape, mode="constant")
        
        if self.p_func == "maxpooling":
            self.A_P = self.X.reshape(self.Shape_P).max(axis=(1, 3))
        elif self.p_func == "averagepooling":
            self.A_P = self.X.reshape(self.Shape_P).mean(axis=(1, 3))
        return self.A_P
    
    def Backward(self, delta_P):
        delta_X = np.kron(delta_P, np.ones(self.Shape_X))
        if self.p_func == "maxpooling":
            mask = self.X == np.kron(self.A_P, np.ones(self.Shape_X))
            mask = mask * self.max_mat
            maxi = mask.reshape(self.Shape_P).max(axis=(1, 3))
            delta_X *= mask == np.kron(maxi, np.ones(self.Shape_X))
        elif self.p_func == "averagepooling":
            delta_X /= np.prod(self.s_q)
        
        return delta_X[self.k_Left:self.s_Right, self.k_Top:self.s_Bottom, :, :]
    
class Transposed_Conv(Layer):
    # X: s_i1 x s_i2 x n_v x M
    # K: s_f1 x s_f2 x n_v x n_f
    # Z: s_o1 x s_o2 x n_f x M
    
    def __init__(self, n_f, s_f=[3, 3], stride=[2, 2]):
        self.n_f = n_f
        self.s_f = np.array(s_f)
        self.stride = np.array(stride)
        self.b, self.K = None, None
        assert self.n_f > 0 and isinstance(self.n_f, int)
        assert len(s_f) == 2 and len(stride) == 2
        assert np.all(self.s_f > 0) and np.all([isinstance(val, int) for val in s_f])
        assert np.all(self.stride > 0) and np.all([isinstance(val, int) for val in stride])
        
        Total_PAD = np.clip(self.s_f - self.stride, a_min=0, a_max=None).astype(int)
        PAD_Pos = Total_PAD//2
        PAD_Neg = self.s_f - 1 - PAD_Pos
        self.PAD_Forw_shape = tuple(zip(PAD_Neg, PAD_Pos)) + ((0, 0),(0, 0))
        self.PAD_Back_shape = tuple(zip(PAD_Pos, PAD_Neg)) + ((0, 0),(0, 0))
        
    def Param_init(self, input_shape):
        *s_i, self.n_v, self.bs = input_shape
        n_input = tuple(self.s_f) + (self.n_v, self.n_f)
        if self.K is None:
            self.K = np.random.randn(*n_input) * np.sqrt(2 / np.sum(n_input))
        if self.b is None:
            self.b = np.zeros(self.n_f).reshape(-1, 1)
        self.optimizer_K = self.optimizer
        self.optimizer_b = copy_class(self.optimizer)
        
        self.bar_X_shape = tuple(s_i*self.stride) + (self.n_v,)
        self.delta_X_shape = tuple(s_i) + (self.n_v,)
    
    def Forward(self, X):
        self.M = np.shape(X)[-1]
        bar_X = np.zeros((self.bar_X_shape + (self.M,)))
        bar_X[::self.stride[0], ::self.stride[1], :, :] = X
        self.tilde_bar_X = np.pad(bar_X, self.PAD_Forw_shape, "constant")
        
        Z = []
        for r in range(self.n_f):
            conv_X_K = self.b[r][0]
            for h in range(self.n_v):
                expand_K = np.expand_dims(self.K[:,:,h,r], axis=-1)
                conv_X_K += correlate(self.tilde_bar_X[:,:,h,:], expand_K, mode="valid")
            Z.append(conv_X_K)
        return np.moveaxis(Z, 0, 2)
    
    def Backward(self, delta_Z):
        delta_X = np.zeros((self.delta_X_shape + (self.M,)))
        tilde_delta_Z = np.pad(delta_Z, self.PAD_Back_shape, "constant")
        rot_K = np.rot90(self.K, 2, axes=(1, 0))
        for h in range(self.n_v):
            for r in range(self.n_f):
                # Convolution = Cross-correlation + Rotated Kernel
                expand_rot_K = np.expand_dims(rot_K[:,:,h,r], axis=-1)
                corr_dZ_K = correlate(tilde_delta_Z[:,:,r,:], expand_rot_K, mode="valid")
                delta_X[:,:,h,:] += corr_dZ_K[::self.stride[0], ::self.stride[1], :]
        
        dg_K = np.zeros(np.shape(self.K))
        for r in range(self.n_f):
            for h in range(self.n_v):
                corr_X_dZ = correlate(self.tilde_bar_X[:,:,h,:], delta_Z[:,:,r,:], 
                                             mode="valid")
                dg_K[:,:,h,r] += np.squeeze(corr_X_dZ, axis=-1)
            
        dg_K = np.rot90(dg_K, 2, axes=(1,0))
        dg_b = np.sum(delta_Z, axis=(0,1,3))
        self.K += self.optimizer_K.Delta(dg_K)
        self.b += self.optimizer_b.Delta(dg_b).reshape(-1,1)
        return delta_X