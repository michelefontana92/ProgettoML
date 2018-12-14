from Monk import *
from Preprocess_Monk import *
from Bagging import *
import numpy as np

"""
a = np.array([[1, 1], [2, 2], [3, 3]])
print(a)
print(np.insert(a, 1, 5))

print(np.insert(a, 2, 5, axis=1))
"""


#preprocess_monk('monks-1.train','monks-1.test','new-monks-1.test')

X, Y = load_monk("monks-1.train")
X_valid, Y_valid = load_monk("monks-1.test")

"""
mlp = MLP(17,3,1,eta = 0.5,alfa=0.9,use_fan_in=True,range_W_h_start=-0.2,range_W_h_end=0.2,range_W_o_start=-0.7,
          range_W_o_end=0.7,
          activation_hidden="tanh")

mlp.train(X,Y,X_valid,Y_valid,500,True)

plt.plot(mlp.errors_list, label='Training Error',ls="-")
plt.plot(mlp.valid_errors_list, label='Validation Error')
plt.title('Monk1')
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':14})
plt.show()

plt.plot(mlp.accuracies_list, label='Training Accuracy',ls="-")
plt.plot(mlp.valid_accuracies_list, label='Validation Accuracy')
plt.title('Monk1')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':14})
plt.show()
"""


bag = Bag(3,17,3,1,eta = 0.9,alfa=0.9,use_fan_in=True,range_W_h_start=-0.2,range_W_h_end=0.2,range_W_o_start=-0.1,
          range_W_o_end=0.1,
          activation_hidden="tanh")

bag.train(X,Y,X_valid,Y_valid,500,True)

fig = plt.figure()
st = plt.suptitle("Monk 1")
plt.subplot(2,1,1)
plt.plot(bag.bag_errors_list, label='Training Error',ls="-")
plt.plot(bag.bag_valid_errors_list, label='Validation Error')
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})

plt.subplot(2,1,2)
plt.plot(bag.bag_accuracies_list, label='Training Accuracy',ls="dashed")
plt.plot(bag.bag_valid_accuracies_list, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})

plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
plt.savefig("monk1_res.jpg")

plt.show()


"""
X = np.eye(6)
idx = np.random.choice(6,12)
Y = X[idx]
print(X)
print(Y)

print(math.ceil(6*80/100))
"""

