from Monk import *
from Preprocess_Monk import *
from Bagging import *

#preprocess_monk('monks-2.train','monks-2.test','new-monks-2.test')
X, Y = load_monk("monks-2.train")
X_valid, Y_valid = load_monk("monks-2.test")

bag = Bag(3,17,4,1,eta = 0.9,alfa=0.9,use_fan_in=False,range_W_h_start=-0.6,range_W_h_end=0.6,range_W_o_start=-0.6,
          range_W_o_end=0.6,
          activation_hidden="tanh")

bag.train(X,Y,X_valid,Y_valid,500,True)


for i in range(X_valid.shape[0]):
    prediction = bag.classify(np.reshape(X_valid[i],(X_valid[i].shape[0],-1)),0.5)
    print("%s) Predicted = %1f Target %1f"%(i+1, prediction,np.reshape(Y_valid[i],(Y_valid[i].shape[0],-1))))

fig = plt.figure()
st = plt.suptitle("Monk 2")
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



plt.savefig("monk2_res.jpg")
plt.show()

