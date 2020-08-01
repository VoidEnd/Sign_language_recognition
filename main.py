import os
import argparse
from keras import optimizers
from nn_model import neural_network
from keras2tf import keras2tf
from extract_data import extract_data


def train_nn(dataset_loc, train_bool, num_test):
    if (train_bool):
        DATASET_SIZE = 27455
        BATCHSIZE = 32
        EPOCHS = 3
        LEARN_RATE = 0.0001
        DECAY_RATE = 1e-6
        NUM_IMAGES = 10

        training_dataset_filepath = '%smnist/sign_mnist_train/sign_mnist_train.csv' % dataset_loc
        testing_dataset_filepath = '%smnist/sign_mnist_test/sign_mnist_test.csv' % dataset_loc

        train_data, train_label, val_data, val_label, testing_data, testing_label = extract_data(
            training_dataset_filepath, testing_dataset_filepath, num_test)

        model = neural_network()

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=LEARN_RATE, decay=DECAY_RATE),
                      metrics=['accuracy']
                      )

        model.fit(train_data,
                  train_label,
                  batch_size=BATCHSIZE,
                  shuffle=True,
                  epochs=EPOCHS,
                  validation_data=(val_data, val_label)
                  )

        scores = model.evaluate(testing_data,
                                testing_label,
                                batch_size=BATCHSIZE
                                )

        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])

        model.save(os.path.join('./train', 'keras_trained_model.h5'))
        print('Finished Training')

    print('Convert Keras to TF')
    keras2tf('train/keras_trained_model.h5', 'train/tfchkpt.ckpt', 'train')


def main():
    argpar = argparse.ArgumentParser()

    argpar.add_argument('--dataset',
                        type=str,
                        default='./',
                        help='The directory where the dataset is held')

    argpar.add_argument('--train', dest='train', action='store_true')
    argpar.add_argument('--no-train', dest='train', action='store_false')
    argpar.add_argument('--num_test', dest='num_test', action='store', type=int, )
    argpar.set_defaults(train=True)
    argpar.set_defaults(num_test=10)

    args = argpar.parse_args()

    train_nn(args.dataset, args.train, args.num_test)


if __name__ == "__main__":
    main()
