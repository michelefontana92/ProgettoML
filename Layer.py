import numpy as np
from Unit import *

class Layer:

    @property
    def n_units(self):
        return self._n_units

    @n_units.setter
    def n_units(self,value):
        self._n_units = value

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self,value):
        self._units = value

    @property
    def layer_output(self):
        return self._layer_output

    @layer_output.setter
    def layer_output(self,value):
        self._layer_output = value

    @property
    def layer_delta(self):
        return self._layer_delta

    @layer_delta.setter
    def layer_delta(self, value):
        self._layer_delta = value

    @property
    def layer_net(self):
        return self._layer_net

    @layer_net.setter
    def layer_net(self, value):
        self._layer_net = value

    @property
    def layer_gradient(self):
        return self._layer_gradient

    @layer_gradient.setter
    def layer_gradient(self, value):
        self._layer_gradient = value

    def __init__(self, n_units):

        self._n_units = n_units
        self._units = []
        self._layer_output = np.zeros((self._n_units,1))
        self._layer_delta = np.zeros((self._n_units,1))
        self._layer_net = np.zeros((self._n_units,1))
        self._layer_gradient = np.zeros((self._n_units,1))

    def compute_layer_output(self,x,weights):
        assert x.shape[1] == 1
        assert x.shape[0] == weights.shape[1]

        for (i, unit) in enumerate(self._units):

            wi = np.reshape(np.array(weights[i]),(weights.shape[1], -1))
            self._layer_output[i] = unit.compute_unit_output(wi, x)
            self._layer_net[i] = unit.net

        return self._layer_output

    def compute_layer_gradient(self):

        for(i, unit) in enumerate(self._units):
            self._layer_gradient[i] = unit.compute_unit_gradient()

        return self._layer_gradient

    def update_bias(self, new_bias):

        for (i,unit) in enumerate(self._units):
            unit.bias = unit.bias + new_bias


class HiddenLayer(Layer):

    def __init__(self, n_units,activation_type):
        super(HiddenLayer,self).__init__(n_units)

        for i in range(0, n_units):
            self.units.append(HiddenUnit(activation_type))

    def compute_layer_delta(self, weights_up, delta_up):

        assert weights_up.shape[1] == self.n_units

        assert delta_up.shape[0] == weights_up.shape[0]

        assert delta_up.shape[1] == 1

        for (i, unit) in enumerate(self.units):

            wi = np.reshape(np.array(weights_up[:,i]), (weights_up.shape[0], -1))
            d = unit.compute_unit_delta(wi,delta_up)
            self.layer_delta[i] = d
            self.layer_gradient[i] = unit.gradient

        return self.layer_delta


class OutputLayer(Layer):

    def __init__(self, n_units,activation_type):
        super(OutputLayer,self).__init__(n_units)

        for i in range(0, n_units):
            self.units.append(OutputUnit(activation_type))

    def compute_layer_delta(self, target):

        assert target.shape[1] == 1
        assert target.shape[0] == self.layer_output.shape[0]

        for(i, unit) in enumerate(self.units):

            d = unit.compute_unit_delta(target[i])

            self.layer_delta[i] = d
            self.layer_gradient[i] = unit.gradient

        return self.layer_delta


"""
W = np.array([[0,0,0],[1,2,3]])
x = np.array([[1,2,3]])
W_up = np.array([[1,2],[1,2],[1,2]])
delta_up = np.array([[1,2,3]])
print("W=",W)
print("x",x)
print("W_up=",W_up)
print("delta_up=",delta_up)
a = np.reshape(np.array(W[0]),(3,-1))
print(a.shape)
layer = HiddenLayer(2)
out = layer.compute_layer_output(x.T,W)
print('out=',out)
delta = layer.compute_layer_delta(W_up,delta_up.T)
print('delta=',delta)
print('gradient=',layer.layer_gradient)

print("\n\n")
x = np.array([[1,2]])
target=np.array([[0.99330715,0.5,0.3]])
print("W=",W_up)
print("x=",x)
print('target=',target.T)
layer_out = OutputLayer(3)
out = layer_out.compute_layer_output(x.T,W_up)
print('out=',out)
delta = layer_out.compute_layer_delta(target.T)
print('layer delta=',delta)
print('layer gradient=',layer_out.layer_gradient)
"""
