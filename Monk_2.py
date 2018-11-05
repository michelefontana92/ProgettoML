from Monk import *

mlp = MLP(17,2,1,eta = 0.5,alfa=0.9,use_fan_in=True,range_W_h_start=-0.7,range_W_h_end=0.7,range_W_o_start=-0.7,range_W_o_end=0.7 )
mlp.train(X,Y,X_valid,Y_valid,1000)
plt.plot(mlp.errors_list, label='Training Error',ls="-")
plt.plot(mlp.valid_errors_list, label='Validation Error')
plt.title('Monk2')
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':14})
plt.show()