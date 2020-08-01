import numpy as np

testing_data = './mnist/sign_mnist_test/sign_mnist_test.csv'
calib_batch_size = 32

testing_data = np.genfromtxt(testing_data, delimiter=',')
testing_data = np.delete(testing_data, 0, 0)
testing_label = testing_data[:, 0]
testing_data = np.delete(testing_data, 0, 1)

np.savetxt('./quantize/quant_calib.csv', testing_data, delimiter=',')
calib_image_data = './quantize/quant_calib.csv'


def calib_input(iter):
    data = np.loadtxt('./quantize/quant_calib.csv', delimiter=',')
    current_iteration = iter * calib_batch_size
    batch_data = data[current_iteration:current_iteration + calib_batch_size]
    batch_data = batch_data.reshape(calib_batch_size, 28, 28, 1).astype('float32') / 255
    return {"input_1_1": batch_data}


calib_input(1)
