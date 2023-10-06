try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class Embedding():
    """
    Embedding layer 추가
    ---
        Args:
            'input_dim' (int): vocabulary 크기와 동일
            'output_dim' (int): layer의 node 개수와 동일 (vector size)
        Returns:
            input: (batch_size, input_length) 크기의 데이터
            output: (batch-size, input_length, output_dim) 크기의 데이터
    """
    def __init__(self, input_dim, output_dim, data_type=np.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = None
        
        self.optimizer = None
        self.data_type = data_type

        self.build()

    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def build(self):
        self.w = np.random.normal(0, pow(self.input_dim, -0.5), (self.input_dim, self.output_dim)).astype(self.data_type)

        self.v, self.m = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)


    def prepare_labels(self, batch_labels):
        batch_labels = batch_labels.astype(np.int32)

        prepared_batch_labels = np.zeros((batch_labels.size, self.input_dim))
        prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1

        return prepared_batch_labels.reshape(self.batch_size, self.current_input_length, self.input_dim).astype(self.data_type)
    

    def forward(self, input_data):
        self.input_data = input_data # (batch_size, input_length)의 크기

        if not all([np.equal(len(self.input_data[0]), len(arr)).all() for arr in self.input_data]):
            raise ValueError("Input sequences must be of the same length")
        
        self.current_input_length = len(self.input_data[0])
        self.batch_size = len(self.input_data)

        self.input_data = self.prepare_labels(self.input_data)
        self.output_data = np.dot(self.input_data, self.w)

        return self.output_data
    

    def backward(self, error):
        self.grad_w = np.matmul(np.transpose(self.input_data, axes=(0, 2, 1)), error).sum(axis=0)

    
    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat = self.optimizer.update(
            self.grad_w,
            self.w,
            self.v,
            self.v_hat,
            self.m_hat,
            layer_num
        )
        return layer_num + 1
    

    def get_grads(self):
        return self.grad_w, self.grad_b
    

    def set_grads(self, grads):
        self.grad_w, self.grad_b = grads
