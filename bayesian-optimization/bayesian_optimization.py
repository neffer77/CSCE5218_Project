from bayes_opt import BayesianOptimization, UtilityFunction
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from EyeStateClassifier import EyeStateClassifier
from sklearn.gaussian_process.kernels import Matern
import tensorflow as tf
count = 0




def cnn_model(num_filters, kernel_size, pool_size, dense_size, learning_rate, batch_size, activation):
    # Convert continuous variables to discrete as needed
    num_filters = int(num_filters)
    kernel_size = int(kernel_size)
    pool_size = int(pool_size)
    dense_size = int(dense_size)
    batch_size = int(batch_size)

    # Choose activation function
    if activation < 0.5:
        activation = 'relu'
    else:
        activation = 'relu'
    global classifier
    classifier = EyeStateClassifier('/Users/connor.neff/OneDrive - ServiceNow/CSCE 5218/Driver Drowsiness Dataset (DDD)/train/', '/Users/connor.neff/OneDrive - ServiceNow/CSCE 5218/Driver Drowsiness Dataset (DDD)/test/',num_filters,kernel_size,pool_size,dense_size,learning_rate, batch_size,activation)
    classifier.load_and_preprocess_data()
    classifier.create_model()
    classifier.train_model()
    global count
    count = count +1

    # Save the model
    classifier.save_model(f'model_iteration_{count}.keras')
    return classifier.evaluate_model()

# Bounded region of parameter space
pbounds = {
    'num_filters': (16, 128),
    'kernel_size': (3, 5),
    'pool_size': (2, 3),
    'dense_size': (64, 128),
    'learning_rate': (1e-4, 1e-2),
    'batch_size': (16, 32),
    'activation': (0, 1),  # Representing a choice between two activation functions
}

optimizer = BayesianOptimization(
    f=cnn_model,
    pbounds=pbounds,
    random_state=1,
)

number_of_iterations = 1
for i in range(number_of_iterations):
    optimizer.maximize(
        init_points=0,
        n_iter=2
    )
