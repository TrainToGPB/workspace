from transformer.activations import ReLU
from transformer.layers.base.dense import Dense
from transformer.layers.base.dropout import Dropout


class PositionwiseFeedForward():
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1):
        self.fc1 = Dense(inputs_num=d_model, units_num=d_ff)
        self.activation = ReLU()
        self.fc2 = Dense(inputs_num=d_ff, units_num=d_model)
        self.dropout=Dropout(dropout_rate)


    def forward(self, input_data, training=True):
        input_data = self.fc1.forward(input_data)
        input_data = self.activation.forward(input_data)
        input_data = self.dropout.forward(input_data, training)
        input_data = self.fc2.forward(input_data)
        return input_data
    

    def backward(self, error):
        error = self.fc2.backward(error)
        error = self.dropout.backward(error)
        error = self.activation.backward(error)
        error = self.fc1.backward(error)
        return error
    

    def set_optimizer(self, optimizer):
        self.fc1.set_optimizer(optimizer)
        self.fc2.set_optimizer(optimizer)


    def update_weights(self, layer_num):
        layer_num = self.fc1.update_weights(layer_num)
        layer_num = self.fc2.update_weights(layer_num)
        return layer_num
    