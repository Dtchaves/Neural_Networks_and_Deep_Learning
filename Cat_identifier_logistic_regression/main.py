import numpy as np
import h5py
import matplotlib.pyplot as plt
import modules.help_functions as hf
import modules.logistic_regresion as lr


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = hf.load_dataset()
train_set_x = hf.normalize_data(train_set_x_orig)
test_set_x = hf.normalize_data(test_set_x_orig)

model_results = lr.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.5, print_cost = False)

print("train accuracy: {} %".format(100 - np.mean(np.abs(model_results['Y_prediction_train'] - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(model_results["Y_prediction_test"] - test_set_y)) * 100))
index = 10
plt.imshow(test_set_x[:,index].reshape((test_set_x_orig.shape[1],test_set_x_orig.shape[2], 3)))
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(model_results["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

costs = np.squeeze(model_results['costs'])
plt.plot(costs)
print(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(model_results["learning_rate"]))
plt.show()