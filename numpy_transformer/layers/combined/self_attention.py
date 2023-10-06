try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from transformer.layers.base.dense import Dense
from transformer.layers.base.dropout import Dropout
from transformer.activations import Sigmoid, Softmax


class MultiHeadAttention:
    def __init__(self, d_model=512, heads_num=8, dropout=0.1, data_type=None):
        self.d_model = d_model
        self.heads_num = heads_num

        self.data_type = data_type

        self.d_k, self.d_q, self.d_v = self.d_model // heads_num, self.d_model // heads_num, self.d_model // heads_num
        self.scale = np.sqrt(self.d_k).astype(self.data_type)

        self.K_linear = Dense(
            inputs_num=self.d_model,
            units_num=self.d_k * heads_num,
            use_bias=False,
            data_type=self.data_type
        ) # self.W_K = np.random.randn(self.d_model, self.d_k)
        self.Q_linear = Dense(
            inputs_num=self.d_model,
            units_num=self.d_k * heads_num,
            use_bias=False,
            data_type=self.data_type
        ) # self.W_Q = np.random.randn(self.d_model, self.d_q)
        self.V_linear = Dense(
            inputs_num=self.d_model,
            units_num=self.d_k * heads_num,
            use_bias=False,
            data_type=self.data_type
        ) # self.W_V = np.random.randn(self.d_model, self.d_v)
        self.O_linear = Dense(
            inputs_num=self.d_model,
            units_num=self.d_k * heads_num,
            use_bias=False,
            data_type=self.data_type
        ) # self.W_O = np.random.randn(self.d_model, self.heads_num * self.d_v)

        self.activation = Softmax()
        self.dropout = Dropout(dropout)

    
    def split_heads_forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, self.d_k).transpose(0, 2, 1, 3)
    

    def split_heads_backward(self, x):
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_k)
    

    def group_heads_forward(self, x):
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_k)
    

    def group_heads_backward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, self.d_k).transpose(0, 2, 1, 3)
    

    def forward(self, query, key, value, mask, training=True):
        self.key_len = key.shape[1] # (batch_size, key_len, hidden_dim)
        self.query_len = query.shape[1] # (batch_size, query_len, hidden_dim)
        self.value_len = value.shape[1] # (batch_size, value_len, hidden_dim)

        K = self.K_linear.forward(key)
        Q = self.Q_linear.forward(query)
        V = self.V_linear.forward(value)

        self.K = self.split_heads_forward(K)
        self.Q = self.split_heads_forward(Q)
        self.V = self.split_heads_forward(V)

        energy = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / self.scale

        self.mask = np.asarray(mask)
        if self.mask is not None:
            self.mask = self.mask[:, np.newaxis, ...]
            energy = np.where(self.mask == 0, float('-inf'), energy)

        attention = self.activation.forward(energy)
        
        self.dropout_attention = self.dropout.forward(attention, training)
        output = np.matmul(self.dropout_attention, self.V)

        concat_output = self.group_heads_forward(output)

        O = self.O_linear.forward(concat_output)

        return O, attention
    

    def backward(self, error):
        error = self.O_linear.backward(error)
        error = self.group_heads_backward(error)

        V_error = np.matmul(self.dropout_attention.transpose(0, 1, 3, 2), error)
        
        error = np.matmul(error, self.V.transpose(0, 1, 3, 2))
        error = self.dropout.backward(error)
        error = self.activation.backward(error)

        if self.mask is not None:
            error = np.where(self.mask == 0, 0, error)

        error /= self.scale

        Q_error = np.matmul(error, self.K)
        K_error = np.matmul(self.Q.transpose(0, 1, 3, 2), error).transpose(0, 1, 3, 2)

        V_error = self.split_heads_backward(V_error)
        Q_error = self.split_heads_backward(Q_error)
        K_error = self.split_heads_backward(K_error)

        V_error = self.V_linear.backward(V_error)
        Q_error = self.V_linear.backward(Q_error)
        K_error = self.V_linear.backward(K_error)

        return Q_error, K_error, V_error

