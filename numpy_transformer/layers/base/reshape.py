try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class Reshape():
    def __init__(self, shape) -> None:
        self.shape = shape
        self.input_shape = None

    
    def build(self):
        self.output_shape = self.shape


    def forward_prop(self, input_data):
        self.prev_shape = input_data.shape
        return input_data.reshape(self.prev_shape[0], *self.shape)
    

    def backward_prop(self, error):
        return error.reshape(self.prev_shape)
