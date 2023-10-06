try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class Activation(object):
    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad):
        x = self.x
        f_x = self.forward(x)
        return grad * (f_x * (1.0 - f_x)).astype(x.dtype)
    

class Tanh(Activation):
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad):
        x = self.x
        f_x = self.forward(x)
        return grad * (1.0 - np.power(f_x, 2)).astype(x.dtype)
    

class Softmax(Activation):
    def __init__(self) -> None:
        self.axis = -1

    def forward(self, x):
        self.x = x
        e_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        self.softmax = e_x / np.sum(e_x, axis=self.axis, keepdims=True)
        return self.softmax

    def backward(self, grad=None):
        batch_size = self.x.shape[0]
        softmax = self.forward(self.x)
        J = softmax[..., np.newaxis] * np.tile(
            np.identity(softmax.shape[-1], dtype=self.x.dtype), 
            (softmax.shape[0], *tuple(np.ones(softmax.ndim, dtype=np.int8).tolist()))) \
                - (softmax[..., np.newaxis, :].transpose(*tuple(np.arange(0, softmax.ndim-1, 1, dtype=np.int8).tolist()), -1, -2) @ softmax[..., np.newaxis, :]
        )
        input_grad = grad[..., np.newaxis, :] @ J

        return input_grad.reshape(self.x.shape) / batch_size
    

class LogSoftmax(Activation):
    def __init__(self) -> None:
        self.axis = -1

    def softmax_forward(self, x):
        e_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        self.softmax = e_x / np.sum(e_x, axis=self.axis, keepdims=True)
        return self.softmax

    def forward(self, x):
        self.x = x
        self.log_softmax = np.log(self.softmax_forward(x))
        return self.log_softmax
    
    def backward(self, grad=None):
        batch_size = self.x.shape[0]
        softmax = self.softmax_forward(self.x)

        input_grad = grad - softmax * grad.sum(axis=self.axis, keepdims=True)

        return input_grad / batch_size
    

class Swish(Activation):
    def __init__(self, beta=1):
        self.beta = beta

    def forward(self, x):
        self.x = x
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z))
        return x * self.sigmoid(self.beta * x)
    
    def backward(self, grad):
        x = self.x
        f_x = self.forward(x)
        return grad * (self.beta * f_x + self.sigmoid(self.beta * x) * (1 - self.beta * f_x).astype(x.dtype))


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, 0, 1).astype(x.dtype)
    

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * x, x)
    
    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, self.alpha, 1).astype(x.dtype)
    

class ELU(Activation):
    def __init__(self, alpht=0.1):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * (np.exp(x) - 1), x)
    
    def backward(self, x):
        x = self.x
        f_x = self.forward(x)
        return grad * np.where(x <= 0, self.alpha + f_x, 1).astype(x.dtype)


class GELU(Activation):
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def backward(self, grad):
        x = self.x
        sech = lambda z: 2 / (np.exp(z) + np.exp(-z))
        return grad * (
            0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + 0.0535161 * np.power(x, 3)
            + 0.398942 * x
            * np.power(sech(0.0356774 * np.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        ).astype(x.dtype)
    

class Identity(Activation):
    def forward(self, x):
        self.x = x
        return x
    
    def backward(self, grad):
        x = self.x
        return np.asarray(grad * np.ones(x.shape).astype(x.dtype))


activations = {
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "swish": Swish(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "elu": ELU(),
    "gelu": GELU(),
    None: Identity()
}
