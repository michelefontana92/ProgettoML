import numpy as np
from Layer import *
import matplotlib.pyplot as plt


class MLP:

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_output(self):
        return self._n_output

    @property
    def n_input(self):
        return self._n_input

    @n_input.setter
    def n_input(self,value):
        self._n_input = value

    @property
    def weight_hidden(self):
        return self._W_h

    @weight_hidden.setter
    def weight_hidden(self,value):
        assert value.shape == self._W_h.shape
        self._W_h = value

    @property
    def weight_output(self):
        return self._W_o

    @weight_output.setter
    def weight_output(self, value):
        assert value.shape == self._W_o.shape
        self._W_o = value

    @property
    def learning_rate(self):
        return self._eta

    @learning_rate.setter
    def learning_rate(self,value):
        self._eta = value

    @property
    def l2_param(self):
        return self._lambda

    @l2_param.setter
    def l2_param(self,value):
        self._lambda = value

    @property
    def momentum(self):
        return self._alfa

    @momentum.setter
    def momentum(self,value):
        self._alfa = value

    @property
    def hidden_layer(self):
        return self._hidden_layer

    @hidden_layer.setter
    def hidden_layer(self,value):
        self._hidden_layer = value

    @property
    def output_layer(self):
        return self._output_layer

    @output_layer.setter
    def output_layer(self, value):
        self._output_layer = value

    @property
    def errors_list(self):
        return self._errors

    @errors_list.setter
    def errors_list(self,value):
        self._errors = value

    @property
    def valid_errors_list(self):
        return self._valid_errors

    @valid_errors_list.setter
    def valid_errors_list(self, value):
        self._valid_errors = value

    @property
    def W_old_hidden(self):
        return self._W_old_hidden

    @W_old_hidden.setter
    def W_old_hidden(self,value):
        assert self._W_old_hidden.shape == value.shape
        self._W_old_hidden = value

    @property
    def W_old_output(self):
        return self._W_old_output

    @W_old_output.setter
    def W_old_output(self, value):
        assert self._W_old_output.shape == value.shape
        self._W_old_output = value

    def weight_initializer(self,range_W_h_start, range_W_h_end,
                 range_W_o_start, range_W_o_end,use_fan_in):

        if use_fan_in:

            self._W_h = np.random.uniform(range_W_h_start / self._n_input,range_W_h_end / self._n_input,
                                          (self._W_h.shape[0],self._W_h.shape[1]))

        else:
            self._W_h = np.random.uniform(range_W_h_start, range_W_h_end,
                                          (self._W_h.shape[0], self._W_h.shape[1]))

        self._W_o = np.random.uniform(range_W_o_start, range_W_o_end,
                                          (self._W_o.shape[0], self._W_o.shape[1]))

    def __init__(self,n_input,n_hidden,n_output,eta=0.1,lambd=0.0,alfa=0.0,range_W_h_start = -0.7, range_W_h_end = 0.7,
                 range_W_o_start=-0.7, range_W_o_end=0.7,use_fan_in=False):

        self._n_hidden = n_hidden
        self._n_output = n_output
        self._n_input = n_input
        self._W_h = np.zeros((n_hidden,n_input))
        self._W_o = np.zeros((n_output,n_hidden))
        self._eta = eta
        self._lambda = lambd
        self._alfa = alfa
        self._hidden_layer = None
        self._output_layer = None
        self._W_old_hidden = np.zeros((n_hidden,n_input))
        self._W_old_output = np.zeros((n_output, n_hidden))
        self._errors = []
        self._valid_errors = []

        self.weight_initializer(range_W_h_start,range_W_h_end,range_W_o_start,range_W_o_end,use_fan_in)

        self._hidden_layer = HiddenLayer(self._n_hidden)
        self._output_layer = OutputLayer(self._n_output)

    def compute_error_pattern_norm(self,output,target):
        assert output.shape[1] == 1
        assert target.shape[1] == 1
        assert output.shape[0] == target.shape[0]

        error = np.linalg.norm(target - output,2)**2

        return 0.5 * error

    def feedforward(self,x):
        assert x.shape[1] == 1
        assert x.shape[0] == self._n_input
        hidd_out = self.hidden_layer.compute_layer_output(x,self.weight_hidden)

        out = self.output_layer.compute_layer_output(hidd_out,self.weight_output)

        return out

    def backpropagation(self,x):

        assert x.shape[1] == 1
        assert x.shape[0] == self.n_input

        deltaW_output = np.zeros((self.weight_output.shape[0],self.weight_output.shape[1]))

        output_layer_delta = self.output_layer.layer_delta
        hidden_layer_delta = self.hidden_layer.layer_delta
        hidden_layer_output = self.hidden_layer.layer_output

        for i in range(self.n_output):
            for j in range(self.n_hidden):
                deltaW_output[i][j] = output_layer_delta[i] * hidden_layer_output[j]

        deltaW_hidden = np.zeros((self.weight_hidden.shape[0], self.weight_hidden.shape[1]))

        for i in range(self.n_hidden):
            for j in range(self.n_input):
                deltaW_hidden[i][j] = hidden_layer_delta[i] * x[j]

        return deltaW_output, deltaW_hidden

    def train(self,X,Y,X_valid = None, Y_valid = None,n_epochs = 1000):

        if X_valid is not None:
            assert Y_valid is not None

        assert X.shape[0] == Y.shape[0]

        n_examples = X.shape[0]
        n_valid_examples = 0
        if X_valid is not None:
            n_valid_examples = X_valid.shape[0]

        for epoch in range(n_epochs):
            error = 0
            err_valid = 0
            deltaW_out = np.zeros(self.weight_output.shape)
            deltaW_hidd = np.zeros(self.weight_hidden.shape)

            for i in range (n_examples):

                x = np.reshape(X[i],(X.shape[1],-1))
                target = np.reshape(Y[i],(Y.shape[1],-1))

                out = self.feedforward(x)

                error += self.compute_error_pattern_norm(out,target)


                delta_out = self.output_layer.compute_layer_delta(target)
                self.hidden_layer.compute_layer_delta(self.weight_output,delta_out)

                dW_out, dW_hidd = self.backpropagation(x)

                deltaW_out = deltaW_out + dW_out
                deltaW_hidd = deltaW_hidd + dW_hidd

            error = error / n_examples
            self.errors_list.append(error)

            if X_valid is not None:

                for i in range(n_valid_examples):
                    x_valid_i = np.reshape(X_valid[i], (X_valid[i].shape[0], -1))
                    y_valid_i = np.reshape(Y_valid[i], (Y_valid[i].shape[0], -1))
                    out_valid = self.feedforward(x_valid_i)
                    err_valid += self.compute_error_pattern_norm(out_valid, y_valid_i)

                err_valid = err_valid / n_valid_examples
                self._valid_errors.append(err_valid)

            if X_valid is not None:
                print("Epoch %s/%s: Train Error: %s Validation Error %s" % (epoch + 1, n_epochs, error,err_valid))

            else:
                print("Epoch %s/%s: Error: %s" % (epoch+1,n_epochs,error))

            deltaW_out = deltaW_out / n_examples
            deltaW_hidd = deltaW_hidd / n_examples

            new_delta_W_output = (self._eta * deltaW_out) + (self._alfa * self._W_old_output)
            self.weight_output = self.weight_output + new_delta_W_output - (self._lambda * self.weight_output)

            new_delta_W_hidden = (self._eta * deltaW_hidd) + (self._alfa * self._W_old_hidden)
            self.weight_hidden = self.weight_hidden + new_delta_W_hidden - (self._lambda * self.weight_hidden)

            self.W_old_hidden = new_delta_W_hidden
            self.W_old_output = new_delta_W_output

        return self.weight_output, self.weight_hidden


"""
print("W_hidden=",mlp.weight_hidden)
print("W_output=",mlp.weight_output)
#mlp.weight_hidden = np.zeros((3,3))
#print(mlp.weight_hidden)
X = np.random.rand(3,1)
out = mlp.hidden_layer.compute_layer_output(X,mlp.weight_hidden)
print("X=",X)
print("Hidden Out=",out)
print("Layer Out",mlp.hidden_layer.layer_output)
out *= 20
print("Layer Out",mlp.hidden_layer.layer_output)
grad = mlp.hidden_layer.compute_layer_gradient()
print('grad = ',grad)
print('layer_grad=',mlp.hidden_layer.layer_gradient)
grad *=2000
print('layer_grad=',mlp.hidden_layer.layer_gradient)

print("\n")
out = mlp.output_layer.compute_layer_output(X,mlp.weight_output)
print("X=",X)
print("Output Out",out)
print("Layer output",mlp.output_layer.layer_output)
out *= 20
print("Layer output",mlp.output_layer.layer_output)
grad = mlp.output_layer.compute_layer_gradient()
print('grad = ',grad)
print('layer_grad=',mlp.output_layer.layer_gradient)
grad *=2000
print('layer_grad=',mlp.output_layer.layer_gradient)
Y = np.random.rand(2,1)
print("target= ",Y)
delta_out = mlp.output_layer.compute_layer_delta(Y)
print("delta= ",delta_out)
print("layer_delta= ",mlp.output_layer.layer_delta)
delta_hidd = mlp.hidden_layer.compute_layer_delta(mlp.weight_output,mlp.output_layer.layer_delta)
print("delta= ",delta_hidd)
print("layer_delta= ",mlp.hidden_layer.layer_delta)

print("\nFeedForward")
out = mlp.feedforward(X)
print("target= ",Y)
print("FF Out=",out)
print("HL Out = ",mlp.hidden_layer.layer_output)
print("OL Out = ",mlp.output_layer.layer_output)
print("HL Net = ",mlp.hidden_layer.layer_net)
print("OL Net = ",mlp.output_layer.layer_net)
error = mlp.compute_error_pattern(mlp.output_layer.layer_output, Y)
out_error = mlp.compute_error_pattern(mlp.output_layer.layer_output, Y)
print("OL Error=",error)
print("FF Error=",out_error)
error = mlp.compute_error_pattern_norm(mlp.output_layer.layer_output, Y)
print("Norm Error=",error)

print("\nBackPropagation")
deltaW = mlp.backpropagation(X)
print("DWOutput=",deltaW[0])
print("DWHidden=",deltaW[1])
"""


"""
print("\nTrain")
mlp = MLP(3,4,1,use_fan_in=True,eta = 0.4, alfa= 0.8, lambd=0)
X = np.random.rand(1000,3)

Y = np.zeros((1000,1))
for i in range(1000):
    Y[i] = np.sin(np.linalg.norm(X[i],2)**2) + np.tanh( np.sin(np.linalg.norm(X[i],2)**2))

Y += np.random.randn(1000,1)
print(mlp.train(X,Y,100))
plt.plot(mlp.errors_list, label='Training Error',ls="-")
plt.title('Prova')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':14})
plt.show()
"""