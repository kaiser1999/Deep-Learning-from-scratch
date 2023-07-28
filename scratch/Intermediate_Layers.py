import numpy as np
import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))

class Layer:
    def build(self, input_shape):
        self.Param_init(input_shape)
        return self.Forward(np.random.normal(size=input_shape)).shape
    
    def Param_init(self, input_shape):
        self.input_shape = input_shape
    
    def Forward_Test(self, x):
        return self.Forward(x)

class Activation(Layer):
    def __init__(self, a_func):
        self.a_func = a_func.lower()
        self.const = 709.78     # max exp of np.float64 value
        assert self.a_func in ["relu", "sigmoid", "tanh", "softmax", "linear"]
    
    def Forward(self, z):
        if self.a_func == "relu":
            self.a = (z > 0)
            return z * self.a
        elif self.a_func == "sigmoid":
            self.a = 1. / (1. + self.exp_clip(-z))
        elif self.a_func == "tanh":
            exp_z = self.exp_clip(z)
            exp_neg_z = self.exp_clip(-z)
            self.a = (exp_z - exp_neg_z)/(exp_z + exp_neg_z)
        elif self.a_func == "softmax":
            exp_z = self.exp_clip(z)
            self.a = exp_z / np.sum(exp_z, axis=0)
        elif self.a_func == "linear":
            self.a = z
        return self.a
    
    def exp_clip(self, x):
        return np.exp(np.clip(x, a_min=None, a_max=self.const))
    
    def Backward(self, delta_A):
        if self.a_func == "relu":
            return self.a * delta_A
        elif self.a_func == "sigmoid":
            return self.a * (1. - self.a) * delta_A
        elif self.a_func == "tanh":
            return (1. - self.a**2) * delta_A 
        elif self.a_func == "softmax":
            return delta_A      # Already included in Categorical loss
            # Avoid numerical error
            #d_L, M = np.shape(self.a)
            #delta_X = np.array([self.a[:,i].reshape(-1, 1) * 
            #                    (np.eye(d_L) - self.a[:, i]) 
            #                    @ delta_A[:,i] for i in range(M)]).T
            #return delta_X
        elif self.a_func == "linear":
            return delta_A
    
class Dropout(Layer):
    def __init__(self, p, method="binomial"):
        self.p = p
        self.method = method.lower()
        self.n_f = None
        if self.method == "gaussian": self.sd = np.sqrt(self.p*(1-self.p))
        assert 0 < p < 1 and self.method in ["binomial", "gaussian"]
        
    def Param_init(self, input_shape):
        self.n_f = input_shape[-2]
    
    def Forward(self, x):
        if self.method == "binomial":
            self.r = np.random.binomial(1, self.p, size=self.n_f)
        elif self.method == "gaussian":
            self.r = np.random.normal(loc=1, scale=self.sd, size=self.n_f)
        return self.r.reshape(-1, 1) * x

    def Backward(self, delta_x):
        return self.r.reshape(-1, 1) * delta_x
    
    def Forward_Test(self, x):
        if self.method == "binomial":
            return x * self.p
        elif self.method == "gaussian":
            return x
        
class Batch_Normalization(Layer):
    def __init__(self, method="original", epsilon=1e-8, decay=0.9):
        self.epsilon = epsilon
        self.method = method.lower()
        self.decay = decay
        self.beta, self.gamma = None, None
        self.mu_test, self.nu_test = None, None
        assert self.method in ["original", "momentum"]
        assert self.epsilon > 0 and 0 < self.decay < 1
    
    def Param_init(self, input_shape):
        n_f, self.bs = input_shape[-2:]
        if len(input_shape) == 2:
            self.sum_axis, self.K = -1, 1
        elif len(input_shape) == 4:
            self.sum_axis, self.K = (0, 1, 3), np.prod(input_shape[:2])
        
        if self.beta is None:
            self.beta = np.zeros(n_f).reshape(-1, 1)
        if self.gamma is None:
            self.gamma = np.random.randn(n_f).reshape(-1, 1) * np.sqrt(2 / n_f)
        self.optimizer_beta = self.optimizer
        self.optimizer_gamma = copy_class(self.optimizer)
        self.mu, self.nu = [], []

    def Forward(self, x):
        self.M = np.shape(x)[-1]
        mu = np.mean(x, axis=self.sum_axis).reshape(-1, 1)
        nu = np.var(x, axis=self.sum_axis).reshape(-1, 1)
        self.iv = nu + self.epsilon
        self.x_hat = (x - mu)/np.sqrt(self.iv)
        
        self.mu.append(mu)
        self.nu.append(nu)
        return self.x_hat * self.gamma + self.beta
    
    def Backward(self, delta_y):
        dg_beta = np.sum(delta_y, axis=self.sum_axis).reshape(-1, 1)
        dg_gamma = np.sum(delta_y*self.x_hat, axis=self.sum_axis).reshape(-1, 1)
        Term = self.M * delta_y - dg_beta/self.K - dg_gamma/self.K * self.x_hat
        delta_x = Term * self.gamma / np.sqrt(self.iv) / self.M
        
        self.beta += self.optimizer_beta.Delta(dg_beta)
        self.gamma += self.optimizer_gamma.Delta(dg_gamma)
        return delta_x
    
    def Inference_init(self):
        if self.method == "original":
            mu_test = np.mean(self.mu, axis=0)
            nu_test = np.mean(self.nu, axis=0) * self.bs / (self.bs - 1)
        elif self.method == "momentum":
            mu_test, nu_test = 0, 0
            for i in range(len(self.mu)):
                mu_test = self.decay * mu_test + (1 - self.decay) * self.mu[i]
                nu_test = self.decay * nu_test + (1 - self.decay) * self.nu[i]
            mu_test = mu_test / (1 - self.decay**len(self.mu))
            nu_test = nu_test / (1 - self.decay**len(self.nu))
        self.mu_test = mu_test.reshape(-1, 1)
        self.nu_test = nu_test.reshape(-1, 1)
    
    def Forward_Test(self, x):
        if self.mu_test is None:
            self.Inference_init()
        x_hat_test = (x - self.mu_test) / np.sqrt(self.nu_test + self.epsilon)
        return x_hat_test * self.gamma + self.beta

class Instance_Noramlization(Layer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.beta, self.gamma = None, None
        assert self.epsilon > 0
    
    def Param_init(self, input_shape):
        n_f, self.sum_axis, self.sum_M_axis = input_shape[-2], (0, 1), (0, 1, 3)
        self.K = np.prod(input_shape[:2])
        if self.beta is None:
            self.beta = np.zeros(n_f).reshape(-1, 1)
        if self.gamma is None:
            self.gamma = np.random.randn(n_f).reshape(-1, 1) * np.sqrt(2 / n_f)
        self.optimizer_beta = self.optimizer
        self.optimizer_gamma = copy_class(self.optimizer)
    
    def Forward(self, x):
        self.M = np.shape(x)[-1]
        mu = np.mean(x, axis=self.sum_axis)
        nu = np.var(x, axis=self.sum_axis)
        self.iv = nu + self.epsilon
        self.x_hat = (x - mu)/np.sqrt(self.iv)
        return self.x_hat * self.gamma + self.beta

    def Backward(self, delta_y):
        dg_beta = np.sum(delta_y, axis=self.sum_M_axis).reshape(-1, 1)
        dg_gamma = np.sum(delta_y*self.x_hat, axis=self.sum_M_axis).reshape(-1, 1)
        
        err_1 = np.sum(delta_y, axis=self.sum_axis)/self.K
        err_2 = np.sum(delta_y * self.x_hat, axis=self.sum_axis)/self.K * self.x_hat
        delta_x = (delta_y - err_1 - err_2) * self.gamma / np.sqrt(self.iv)
        
        self.beta += self.optimizer_beta.Delta(dg_beta)
        self.gamma += self.optimizer_gamma.Delta(dg_gamma)
        return delta_x