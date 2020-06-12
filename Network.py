from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, ReLU


class DQN:
    def __init__(self, lr):
        self.model = self._create_model(lr)

    def _create_model(self, lr):
        input_shape = Input(shape=(3, 3, 4))

        conv = Conv2D(filters=16, kernel_size=2, strides=1, padding="same")(input_shape)
        conv = ReLU()(conv)
        conv = Conv2D(filters=32, kernel_size=2, strides=1, padding="same")(conv)
        conv = ReLU()(conv)

        flatten = Flatten()(conv)

        dense = Dense(units=384, kernel_initializer='random_normal', use_bias=False)(flatten)
        dense = ReLU()(dense)
        action = Dense(units=4, kernel_initializer='random_normal', use_bias=False)(dense)
        model = Model(inputs=input_shape, outputs=action)

        adam = Adam(lr=lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, states, target, epochs=1, verbose=0):
        self.model.fit(states, target, epochs=epochs, verbose=verbose)

    def update_target_net(self, policy_net):
        weights = self.model.get_weights()
        policy_weights = policy_net.model.get_weights()
        for i in range(len(policy_weights)):
            weights[i] = policy_weights[i]
        self.model.set_weights(weights)