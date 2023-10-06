try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class Dropout():
    """
    입력 데이터에 dropout 추가
    ---
        Args:
            'rate' (float): 0부터 1 사이의 dropout rate
        Returns:
            output: Dropout이 적용된, input_data와 동일한 크기의 데이터
    """
    def __init__(self, rate=0.1, data_type=np.float32) -> None:
        self.rate = rate
        self.input_shape = None
        self.data_type = data_type


    def build(self):
        self.output_shape = self.input_shape


    def forward(self, input_data, training=True):
        self.mask = 1.0
        if training:
            self.mask = np.random.binomial(
                n=1,
                p=1-self.rate,
                size=input_data.shape,
            ).astype(self.data_type)

        return input_data * self.mask
    

    def backward(self, error):
        return error * self.mask
