import matplotlib.pyplot as plt
import numpy as np 
import modules.help_functions as hf
import modules.logistic_regresion as lr


def test_data(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    index = 44
    #print(train_set_x_orig.shape)
    #print(train_set_x_orig[index])
    #print(classes[0].decode("utf-8"))
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    plt.show()
#tests.test_data(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)

    
def test_initialize_with_zeros():
    dim = 2
    w, b = hf.initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))
    
def test_lr():
    w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
    grads, cost = lr.propagate(w, b, X, Y)
    print (X)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))
    
    params, grads, costs = lr.optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    
    print ("predictions = " + str(lr.predict(w, b, X)))