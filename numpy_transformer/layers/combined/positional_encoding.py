try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from transformer.layers.base.dropout import Dropout


class PositionalEncoding():
    def __init__(self, max_len, d_model, dropout_rate=0.1, data_type=np.float32):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.data_type = data_type

        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, :, :].astype(self.data_type) # (1, max_len, d_model) 크기


    def forward(self, x):
        return x + self.pe[:, :x.shape[1]] # (batch_size, seq_len, d_model) 크기
    

    def backward(self, error):
        return error # (batch_size, seq_len, d_model) 크기
