from MLP import *
from Utils import *
import numpy as np
from matplotlib import pyplot as plt


class Bag:

    @property
    def bag_errors_list(self):
        return self._bag_errors

    @bag_errors_list.setter
    def bag_errors_list(self, value):
        self._bag_errors = value

    @property
    def bag_valid_errors_list(self):
        return self._bag_valid_errors

    @bag_valid_errors_list.setter
    def bag_valid_errors_list(self, value):
        self._bag_valid_errors = value

    @property
    def bag_valid_accuracies_list(self):
        return self._bag_valid_accuracies

    @bag_valid_accuracies_list.setter
    def bag_valid_accuracies_list(self, value):
        self._bag_valid_accuracies = value

    @property
    def bag_accuracies_list(self):
        return self._bag_accuracies

    @bag_accuracies_list.setter
    def bag_accuracies_list(self, value):
        self._bag_accuracies = value

    @property
    def models(self):
        return self._models

    @property
    def n_models(self):
        return self._n_models

    def __init__(self, n_models, n_input,n_hidden,n_output,eta=0.1,lambd=0.0,alfa=0.0,range_W_h_start = -0.7, range_W_h_end = 0.7,
                 range_W_o_start=-0.7, range_W_o_end=0.7,use_fan_in=False,activation_hidden = "sigmoid", activation_output = "sigmoid"):

        self._n_models = n_models

        self._bag_errors = []
        self._bag_valid_errors = []
        self._bag_accuracies = []
        self._bag_valid_accuracies = []

        self._models = []

        for i in range(n_models):
            self.models.append(MLP(n_input,n_hidden,n_output,eta,lambd,alfa,range_W_h_start, range_W_h_end,
                 range_W_o_start, range_W_o_end,use_fan_in,activation_hidden, activation_output))

    def compute_bag_output(self,epoch,n_examples,n_valid_examples = 0):

        out = np.zeros((n_examples,1))
        valid_out = np.zeros((n_valid_examples,1))
        for (i,model) in enumerate(self._models):
            out += np.array(model.output_epochs[epoch])

        if n_valid_examples > 0:
            for (i, model) in enumerate(self._models):
                valid_out += np.array(model.output_epochs_valid[epoch])

        return out / self.n_models, valid_out / self.n_models

    def compute_bag_prediction(self,epoch,n_examples,n_valid_examples,threshold = 0.5):

        predictions = np.zeros((n_examples, 1))
        valid_predictions = None

        if n_valid_examples > 0:
            valid_predictions = np.zeros((n_valid_examples,1))

        outputs = []
        for (i, model) in enumerate(self._models):
            outputs.append(np.array(model.output_epochs[epoch]))

        for i in range(n_examples):
            classes = np.zeros((self._n_models,1))
            for (model_id,output) in enumerate(outputs):
                classes[model_id] = compute_class(np.reshape(output[i],(output[i].shape[0],-1)),threshold)
            predictions[i] = majority_vote(classes)

        if n_valid_examples > 0:
            outputs = []
            for (i, model) in enumerate(self._models):
                outputs.append(np.array(model.output_epochs_valid[epoch]))

            for i in range(n_valid_examples):
                classes = np.zeros((self._n_models, 1))
                for (model_id,output) in enumerate(outputs):
                    classes[model_id] = compute_class(np.reshape(output[i], (output[i].shape[0], -1)), threshold)
                valid_predictions[i] = majority_vote(classes)

        return predictions, valid_predictions

    def train(self,X_train,Y_train,X_val=None,Y_val=None,n_epochs=1000,check_accuracy=False,threshold = 0.5):

        n_examples = X_train.shape[0]
        n_valids = 0

        if X_val is not None:
            n_valids = X_val.shape[0]

        for (i,model) in enumerate(self._models):
            print("Training model number %s / %s"%(i+1,self.n_models))

            model.train(X_train,Y_train,X_val,Y_val,n_epochs,check_accuracy)
            print("-"*50)

        for epoch in range(n_epochs):

            bag_error = 0
            bag_valid_error = 0
            bag_accuracy = 0
            bag_valid_accuracy = 0

            bag_out = self.compute_bag_output(epoch,n_examples,n_valids)

            for (i,out) in enumerate(bag_out[0]):
                target = np.reshape(Y_train[i], (Y_train[i].shape[0], -1))
                out = np.reshape(out,(out.shape[0],-1))
                bag_error +=  compute_error(out,target)

            bag_error /= n_examples
            self._bag_errors.append(bag_error)

            if check_accuracy:
                predictions,valid_predictions = self.compute_bag_prediction(epoch,n_examples,n_valids,threshold)

                for (i,prediction) in enumerate(predictions):

                    target = np.reshape(Y_train[i],(Y_train[i].shape[0],-1))
                    prediction = np.reshape(prediction, (prediction.shape[0], -1))
                    bag_accuracy += compute_accuracy(prediction,target,threshold)

                bag_accuracy /= n_examples
                self._bag_accuracies.append(bag_accuracy)

                if n_valids > 0:

                    for (i, out) in enumerate(bag_out[1]):
                        target = np.reshape(Y_val[i], (Y_val[i].shape[0], -1))
                        out = np.reshape(out, (out.shape[0], -1))
                        bag_valid_error += compute_error(out, target)

                    for (i,prediction) in enumerate(valid_predictions):
                        target = np.reshape(Y_val[i], (Y_val[i].shape[0], -1))
                        prediction = np.reshape(prediction, (prediction.shape[0], -1))
                        bag_valid_accuracy += compute_accuracy(prediction, target, threshold)

                    bag_valid_error /= n_valids
                    self._bag_valid_errors.append(bag_valid_error)

                    bag_valid_accuracy /= n_valids
                    self._bag_valid_accuracies.append(bag_valid_accuracy)

    def classify(self,x,threshold = 0.5):

        assert x.shape[1] == 1

        votes = np.zeros((self._n_models,1))

        for (i,model) in enumerate(self._models):
            votes[i] = model.predict_class(x,threshold)

        return majority_vote(votes)



