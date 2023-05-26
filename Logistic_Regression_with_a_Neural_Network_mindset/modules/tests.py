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
    
def test_model(test_set_x_orig,train_set_x, train_set_y, test_set_x, test_set_y,classes):
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
    
def learning_rates(test_set_x_orig,train_set_x, train_set_y, test_set_x, test_set_y,classes):
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = lr.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()