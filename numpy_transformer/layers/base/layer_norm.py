try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class LayerNormalization():
    """
    입력 데이터에 layer normalization 적용
    ---
        Args:
            'momentum' (float): 이동 평균의 momentum 매개변수
            'epsilon' (float): 알고리즘의 epsilon
        Returns:
            output: normalized된 입력 데이터
    """
    def __init__(self, normalized_shape=None, epsilon=0.001, data_type=np.float32):
        self.normalized_shape = normalized_shape
        self.normalized_axis = None
        self.epsilon = epsilon

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.optimizer = None
        self.data_type = data_type

        self.axis = None

        self.build()


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def build(self):
        self.feature_size = None
        if self.normalized_shape is not None:
            self.gamma = np.ones(self.normalized_shape).astype(self.data_type)
            self.beta = np.zeros(self.normalized_shape).astype(self.data_type)

            self.vg, self.mg = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)
            self.vg_hat, self.mg_hat = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)

            self.vb, self.mb = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)
            self.vb_hat, self.mb_hat = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)


    def forward(self, input_data):
        self.input_data = input_data
        x_T = self.input_data.T

        if self.normalized_shape is None:
            self.normalized_shape = self.input_data.shape[1:]
            self.build()

        self.normalized_axis = tuple(np.arange(self.input_data.ndim - self.gamma.ndim).tolist())

        self.mean = np.mean(x_T, axis=0)
        self.var = np.var(x_T, axit=0)

        self.x_centered = x_T - self.mean
        self.stdv_inv = 1 / np.sqrt(self.var + self.epsilon)

        self.x_hat_T = self.x_centered * self.stdv_inv
        self.x_hat = self.x_hat_T

        self.output_data = self.gamma * self.x_hat + self.beta

        return self.output_data
    

    def backward(self, error):
        error_T = error.T

        output_error = (1 / self.feature_size) \
            * np.expand_dims(self.gamma, axis=self.normalized_axis).T \
            * self.stdv_inv \
            * (
                self.feature_size * error_T \
                - np.sum(error_T, axis=0) \
                - self.x_centered * np.power(self.stdv_inv, 2) \
                * np.sum(error_T * self.x_centered, axis=0)
            )
        
        output_error = output_error.T

        self.grad_gamma = np.sum(error * self.x_hat, axis = self.normalized_axis)
        self.grad_beta = np.sum(error, axis=self.normalized_axis)

        return output_error
    

    def update_weights(self, layer_num):
        self.gamma, self.vg, self.vg_hat, self.mg_hat = self.optimizer.update(
            self.grad_gamma,
            self.gamma,
            self.vg,
            self.vg_hat,
            self.mg_hat,
            layer_num
        )
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(
            self.grad_beta,
            self.beta,
            self.vb,
            self.mg,
            self.vb_hat,
            self.mb_hat,
            layer_num
        )
        return layer_num + 1
    

    def get_grads(self):
        return self.grad_gamma, self.grad_beta
    

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads
