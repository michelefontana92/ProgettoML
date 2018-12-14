from Monk import *
from Preprocess_Monk import *
import pandas as pd

"""
preprocess_monk('monks-3.train','monks-3.test','new-monks-3.test')
X, Y = load_monk("monks-3.train")
X_valid, Y_valid = load_monk("new-monks-3.test")


for i in range(X_valid.shape[0]):
    print("%s) Test Encoding = %s Target= %s" % (i+1, X_valid[i],Y_valid[i]))
mlp = MLP(17,2,1,eta = 0.8,alfa=0.8,lambd=0.005,use_fan_in=True,range_W_h_start=-0.7,range_W_h_end=0.7,range_W_o_start=-0.7,range_W_o_end=0.7,
          activation_hidden="tanh")

mlp.train(X,Y,X_valid,Y_valid,500,True)
plt.plot(mlp.errors_list, label='Training Error',ls="-")
plt.plot(mlp.valid_errors_list, label='Validation Error')
plt.title('Monk3')
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':14})
plt.show()

plt.plot(mlp.accuracies_list, label='Training Accuracy',ls="-")
plt.plot(mlp.valid_accuracies_list, label='Validation Accuracy')
plt.title('Monk3')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':14})
plt.show()

for i in range(X_valid.shape[0]):
    print("%s) Predicted = %s Target = %s" % (i+1,mlp.predict_class(X_valid[i].reshape(X_valid[i].shape[0],-1)), Y_valid[i]))
"""

l = [1,2,3,4]
l1 = [1,2,3,4]

print("sum = ",np.array(l)+np.array(l1))
print("mean = ",np.mean(np.array(l)))
print("variance = ",np.var(np.array(l)))