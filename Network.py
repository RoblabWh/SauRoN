from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, Conv1D, ReLU
from keras.optimizers import Adam


class DQN:
    def __init__(self, lr):
        self.model = self._create_model(lr)

    def _create_model(self, lr):
        # [X Rob] [Y Rob] [lin Vel]
        # [X Goal] [Y Goal] [ang Vel]
        # [lin acc] [ang acc] [orientation]
        input_shape = Input(shape=(4, 7))

        # conv = Conv1D(filters=6, kernel_size=2, strides=1, padding="valid")(input_shape)
        # conv = ReLU()(conv)
        # conv = Conv1D(filters=12, kernel_size=2, strides=1, padding="valid")(conv)
        # conv = ReLU()(conv)
        #
        # flatten = Flatten()(conv)
        flatten = Flatten()(input_shape)

        dense = Dense(units=64, kernel_initializer='random_normal', use_bias=False)(flatten)
        dense = ReLU()(dense)
        action = Dense(units=3, kernel_initializer='random_normal', use_bias=False)(dense)
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
