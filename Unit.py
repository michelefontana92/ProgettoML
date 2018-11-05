import numpy as np


def sigmoid(x):
    return float(1./(1+np.exp(-x)))


def sigmoid_gradient(x):
    return float(x * (1-x))


class Unit:

    @property
    def net(self):
        return self._net;

    @net.setter
    def net(self,value):
        self._net = value;

    @property
    def out(self):
        return self._out;

    @out.setter
    def out(self,value):
        self._out = value;

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self,value):
        self._gradient = value

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self,value):
        self._delta = value

    def __init__(self):
        self._net = 0.0
        self._out = 0.0
        self._delta = 0.0
        self._gradient = 0.0

    def compute_unit_output(self,w,x):
        assert(x.shape[1] == 1)
        assert(x.shape[1] == 1)
        assert (w.shape == x.shape)

        self._net = float(np.dot(w.T, x))
        self._out = sigmoid(self._net);
        return float(self._out)

    def compute_unit_gradient(self):
        self._gradient = sigmoid_gradient(self._out)
        return self._gradient


class OutputUnit(Unit):

    def __init__(self):
        super(OutputUnit,self).__init__();

    def compute_unit_delta(self,target):
        self._gradient = self.compute_unit_gradient()
        self._delta = float(target - self.out) * self.gradient
        return self._delta


class HiddenUnit(Unit):

    def __init__(self):
        super(HiddenUnit,self).__init__()

    def compute_unit_delta(self,w_up,delta_up):

        assert w_up.shape[1] == 1
        assert delta_up.shape[1] == 1
        assert w_up.shape[0] == delta_up.shape[0]

        self._gradient = self.compute_unit_gradient()
        self._delta = float(np.dot(w_up.T,delta_up)* self._gradient)
        return self._delta
"""
X = np.array([[1,2,3]])
W = np.array([[1,2,3]])
print(X.shape)
assert (X.shape == (1,3))
unit = Unit()
print("out=",unit.compute_unit_output(W.T,X.T))
print("gradient",unit.compute_unit_gradient())
print("gradient",unit.gradient)

Y = np.array([[2]])
out_unit = OutputUnit()
out_unit.compute_unit_output(W.T,X.T)
out_unit.compute_unit_delta(Y.T)
print("delta",out_unit.delta)

W_up = np.array([[1,2,3]])
delta_up = np.array([[1,2,3]])

hidden = HiddenUnit()
hidden.compute_unit_output(W.T,X.T)
hidden.compute_unit_delta(W_up.T,delta_up.T)
print("hidden_out",hidden.out)
print("hidden_gradient",hidden.gradient)
print("hidden_delta",hidden.delta)"""













