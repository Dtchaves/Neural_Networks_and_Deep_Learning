import numpy as np 
import h5py
import modules.help_functions as hf

eps = 0.00001

def forward_propagation(w,b,X,Y):
    A = hf.sigmoid(np.dot(w.T,X) + b)
    cost = (1/X.shape[1]) * -(np.sum(Y * np.log(A+eps) + (1 - Y) * np.log(1 - (A-eps))))
    return A,cost

def backward_propagation(w,b,X,Y,A):
    dw = (1/A.size) * (np.dot(X,((A-Y).T)))
    db = np.sum((A - Y)) * (1/A.size)
    return dw,db

def propagate(w,b,X,Y):
    
    A,cost = forward_propagation(w,b,X,Y)
    dw,db = backward_propagation(w,b,X,Y,A)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    
    costs = []
    
    for i in range(num_iterations):
        grands,cost = propagate(w,b,X,Y)
        dw = grands["dw"]
        db = grands["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db
        if not(i % 100):
            costs.append(cost)
            
        if print_cost and not(i % 100):
            #print (f"Cost after iteration {i}: {cost}")
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
        
def predict(w, b, X):
    Y_prediction = np.zeros((1,X.shape[1]))
    
    A = hf.sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0][i] > 0.5:  Y_prediction[0][i] = 1 
        else: Y_prediction[0][i] = 0


    assert(Y_prediction.shape == (1, X.shape[1]))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    
    w, b = hf.initialize_with_zeros(X_train.shape[0])
    
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d