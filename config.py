class DefaultConfig(object):


    learn_frequency = 5
    memory_size = 2500000
    memory_warmup_size = 500
    constraint_network = 500
    batch_size = 64
    learning_rate = 0.0001
    gamma = 0.95

    # Training data
    result_path = r"./res/test"
    model_path = r"./saved_model/test"

    #truth model
    resisty = [2500, 1000, 100, 10, 100, 25, 10, 2.5]
    thickness = [600, 1400, 2200, 3400, 7000, 9000, 11000]
    regularization_factor=0.01
    noise_factor=0.1