import numpy as np
import h5py
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import modules.help_functions as hf
import modules.logistic_regresion as lr


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = hf.load_dataset()
train_set_x = hf.normalize_data(train_set_x_orig)
test_set_x = hf.normalize_data(test_set_x_orig)

model_trained = lr.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.5, print_cost = False)
w = model_trained["w"]
b = model_trained["b"]


img = plt.imread("/home/diogo/Documentos/IC/Neural_Networks_and_Deep_Learning_Exercices/Cat_identifier_logistic_regression/dataset/photo")
res = resize(img, (64, 64,3)) 
plt.imshow(res)
plt.show()


image = res.reshape(1,-1).T
my_predicted_image = lr.predict(w, b, image)
plt.axis('off')
plt.title("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.imshow(img)
plt.show()

