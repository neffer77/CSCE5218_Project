from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from EyeStateClassifier import EyeStateClassifier




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
        activation = 'sigmoid'

    classifier = EyeStateClassifier('../../Driver Drowsiness Dataset (DDD)/train', '../../Driver Drowsiness Dataset (DDD)/train',num_filters,kernel_size,pool_size,dense_size,learning_rate, batch_size,activation)
    classifier.load_and_preprocess_data()
    classifier.create_model()
    classifier.train_model()
    return classifier.evaluate_model()

# Bounded region of parameter space
pbounds = {
    'num_filters': (16, 256),
    'kernel_size': (3, 5),
    'pool_size': (2, 3),
    'dense_size': (128, 512),
    'learning_rate': (1e-4, 1e-2),
    'batch_size': (16, 128),
    'activation': (0, 1),  # Representing a choice between two activation functions
}

optimizer = BayesianOptimization(
    f=cnn_model,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=2,
)
