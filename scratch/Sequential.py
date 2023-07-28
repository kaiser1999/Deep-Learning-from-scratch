import numpy as np
from tqdm import trange
import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))

class Sequential:
    def __init__(self):
        self.network = []
    
    def add(self, layers):
        self.network.append(layers)
    
    def Create_mini_batch(self, X, y=None, bs=32):
        # 2d: N x D; 3d: N x T x D; 4d: N x s_I x s_I x n_v
        if len(np.shape(X)) == 1:       # for Dense with 1 feature variable
            X = np.expand_dims(X, axis=-1)
        X_batch, y_batch = [], []
        for i in range(0, len(X), bs):
            X_batch.append(X[i:i+bs])
            if y is not None:
                y_batch.append(y[i:i+bs])
        return X_batch, y_batch
    
    def Categorical_Cross_Entropy(self, y_one_hot, a):
        M = np.shape(y_one_hot)[-1]
        hat_y = a==np.max(a, axis=0)
        acc = np.sum(y_one_hot * hat_y)
        l = -np.sum(y_one_hot * np.log(np.clip(a, a_min=1e-25, a_max=1)))
        error = (a - y_one_hot) / M # error with softmax
        #error = -y_one_hot / a / M # Original error
        return error, l, acc
    
    def MSE(self, y_train, a):
        M = np.shape(y_train)[-1]
        acc = 0
        l = np.sum((y_train - a)**2)
        error = -2*(y_train - a) / M
        return error, l, acc
        
    def One_hot_label(self, y):
        self.uni_labels = np.unique(y)
        label = np.zeros((len(y), len(self.uni_labels)))
        pos = np.where(self.uni_labels == y.reshape(-1, 1))[1]
        label[np.arange(len(y)), pos] = 1
        return label
    
    def compile(self, optimizer, loss="MSE"):
        self.optimizer = optimizer
        self.loss = loss.lower()
        assert self.loss in ["mse", "categorical_cross_entropy"]
        for layer in self.network:
            layer.optimizer = copy_class(self.optimizer)
    
    def train(self, X, y, bs=32, EPOCHS=1e4, shuffle=True):
        indices = np.arange(len(y))
        if self.loss == "categorical_cross_entropy":
            y = self.One_hot_label(y)
            postfix0 = "Loss: 0.0000, Accuracy: 0.0000"
        else:
            postfix0 = "Loss: 0.0000"
        
        X_train, y_train = self.Create_mini_batch(X[indices], y[indices], bs)
        self.bs = bs
        input_shape = np.moveaxis(X_train[0], 0, -1).shape
        for layer in self.network:
            input_shape = layer.build(input_shape)
        
        add_zero = len(str(int(EPOCHS)))
        for epoch in range(int(EPOCHS)):
            if shuffle:
                indices = np.random.permutation(len(y))
                X_train, y_train = self.Create_mini_batch(X[indices], y[indices], bs)
                
            description = "Epoch: " + f"{epoch+1}".zfill(add_zero) + f"/{EPOCHS}"
            
            t = trange(len(y_train), desc=description, postfix=postfix0)
            Loss, Accuracy = 0, 0
            for i in t:
                a = np.moveaxis(X_train[i], 0, -1) # move m to last index
                for layer in self.network:
                    a = layer.Forward(a)
                if self.loss == "categorical_cross_entropy":
                    err, l, acc = self.Categorical_Cross_Entropy(y_train[i].T, a)
                else:
                    err, l, acc = self.MSE(y_train[i].T, a)
                Accuracy += acc / len(y)
                Loss += l / len(y)
                
                for layer in reversed(self.network):
                    err = layer.Backward(err)
                
                if self.loss == "categorical_cross_entropy":
                    t.set_postfix_str(f"Loss: {Loss:0.4f}, Accuracy: {Accuracy:0.4f}")
                else:
                    t.set_postfix_str(f"Loss: {Loss:0.4f}")
    
    def predict(self, X):
        X_test, y_test = self.Create_mini_batch(X, None, min(self.bs, len(X)))
        y_pred = []
        for i in trange(len(X_test)):
            a = np.moveaxis(X_test[i], 0, -1)
            for layer in self.network:
                a = layer.Forward_Test(a)
            if self.loss == "categorical_cross_entropy":
                hat_y = self.uni_labels[np.argmax(a, axis=0)]
            else:
                hat_y = a.T
            y_pred.append(hat_y)
        return np.concatenate(y_pred, axis=0)