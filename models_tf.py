import math
from abc import ABC, abstractmethod

# import ray.experimental.tf_utils as rtfu
import numpy as np

class MuZeroNetwork:
    def __new__(cls, config, training):
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
                training)

##################################
############# ResNet #############
class MuZeroResidualNetwork:
    def __init__(self,
                 observation_shape,
                 stacked_observations,
                 action_space_size,
                 num_blocks,
                 num_channels,
                 reduced_channels_reward,
                 reduced_channels_value,
                 reduced_channels_policy,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 support_size,
                 downsample,
                 training
                 ):
        import tensorflow as tf

        sess = tf.get_default_session()
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.compat.v1.InteractiveSession()

        self.num_channels = num_channels
        self.support_size = 10
        self.input_shape = observation_shape
        self.num_actions = action_space_size
        self.training = training
        self.num_res_blocks = num_blocks
        self.init = 'he_normal'
        self.input_ph = tf.placeholder(tf.float32, (None,)+self.input_shape, 'input_img')
        self.hidden_ph = tf.placeholder(tf.float32, (None,) + (self.num_channels,
                                                               self.input_shape[0],
                                                               self.input_shape[1]), 'input_hidden')
        self.action_ph = tf.placeholder(tf.float32, (None,) + (self.num_actions,), 'input_action')

        self.representation_model = self.get_representation_model(self.input_shape)
        self.prediction_model = self.get_prediction_model()
        self.dynamics_model = self.get_dynamics_model()

        self.representation = self.representation_model(self.input_ph)
        self.representation_normed = self.normalize_representation(self.hidden_ph)
        self.policy, self.value = self.prediction_model(self.hidden_ph)
        self.support_ph = tf.placeholder(tf.float32, (None, int(2*self.support_size + 1)), 'input_support')
        self.scalar_out = support_to_scalar(self.support_ph, self.support_size)

        self.extended_state = self.concat_state_and_action(self.hidden_ph, self.action_ph)
        self.next_state, self.next_reward = self.dynamics_model(self.extended_state)


        # self.hidden_t,self.policy_t,self.value_t = self.initial_inference_symbolic(self.input_ph, training = training)
        # self.next_hidden_t,self.reward_t, self.next_policy_t, self.next_value_t = self.recurrent_inference_symbolic(self.hidden_ph,
        #                                                                                                   self.action_ph, training = training)
        #
        # self.hidden_t_scalar, self.policy_t_scalar, self.value_t_scalar = self.initial_inference_symbolic_scalar(self.input_ph, training = training)
        # self.next_hidden_t_scalar, self.reward_t_scalar, self.next_policy_t_scalar, self.next_value_t_scalar = self.recurrent_inference_symbolic_scalar(
        #     self.hidden_ph,
        #     self.action_ph, training = training)
        # self.variables = rtfu.TensorFlowVariables(c, sess)
        tmp_w = self.get_weights()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.set_weights(tmp_w)

        self.steps_trained = 0


    def get_representation_model(self,in_shape):
        import tensorflow as tf
        inp = tf.keras.layers.Input(in_shape)
        x = inp
        x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3,3),
                                        strides=(1,1), padding='same', use_bias=False,
                                   kernel_initializer=self.init,data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
        x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        for _ in range(self.num_res_blocks):
            x_in = x
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.Dropout(0.05)(x, training=self.training)
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = x_in + x
            x = tf.keras.layers.Activation('relu')(x)
        h=x
        # h = self.normalize_representation(x)
        model = tf.keras.models.Model(inp, h)
        return model

    def get_prediction_model(self):
        import tensorflow as tf
        inp = tf.keras.layers.Input([self.num_channels,self.input_shape[0], self.input_shape[1]])
        x = inp
        for _ in range(self.num_res_blocks):
            x_in = x
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.Dropout(0.05)(x, training=self.training)
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = x_in + x
            x = tf.keras.layers.Activation('relu')(x)
        value_x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(1,1), strides=(1, 1),
                                         kernel_initializer=self.init,data_format='channels_first')(x)
        policy_x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(1,1), strides=(1, 1),
                                          kernel_initializer=self.init,data_format='channels_first')(x)

        value_x = tf.keras.layers.Flatten()(value_x)
        # value_x = tf.keras.layers.Dropout(0.05)(value_x, training=self.training)
        policy_x = tf.keras.layers.Flatten()(policy_x)
        # policy_x = tf.keras.layers.Dropout(0.05)(policy_x, training=self.training)

        value_x = tf.keras.layers.Dense(int(self.num_channels/2),
                                        kernel_initializer=self.init)(value_x)
        value_x = tf.keras.layers.ELU()(value_x)
        value = tf.keras.layers.Dense(2*self.support_size+1)(value_x)

        policy_x = tf.keras.layers.Dense(int(self.num_channels/2),
                                         kernel_initializer=self.init)(policy_x)
        policy_x = tf.keras.layers.ELU()(policy_x)
        policy = tf.keras.layers.Dense(self.num_actions,
                                       kernel_initializer=self.init)(policy_x)

        model = tf.keras.models.Model(inp, [policy, value])

        return model

    def normalize_representation(self,rep):
        import tensorflow as tf
        min_encoded_state = tf.reduce_min(tf.reduce_min(rep, axis=1, keepdims=True), axis=2, keepdims=True)
        max_encoded_state = tf.reduce_max(tf.reduce_max(rep, axis=1, keepdims=True), axis=2, keepdims=True)

        scale_encoded_state = max_encoded_state - min_encoded_state
        less_mask = tf.cast(tf.less(scale_encoded_state, 1e-5), tf.float32)
        scale_encoded_state += less_mask*1e-5
        rep_normalized = (rep - min_encoded_state) / scale_encoded_state
        return rep_normalized

    def get_dynamics_model(self):
        import tensorflow as tf
        inp = tf.keras.layers.Input([self.num_channels+1,self.input_shape[0],self.input_shape[1]])
        x = inp
        x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                        strides=(1, 1), padding='same', use_bias=False,
                                   kernel_initializer=self.init,data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.1, epsilon=1e-5)(x, training=self.training)
        x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        for _ in range(self.num_res_blocks):
            # x = ResidualBlock(self.num_channels, (1, 1))(x, training = self.training)
            x_in = x
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.Dropout(0.05)(x, training=self.training)
            # x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(3, 3),
                                       strides=(1, 1), padding='same', use_bias=False,
                                       kernel_initializer=self.init,data_format='channels_first')(x)
            x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)(x, training=self.training)
            x = x_in + x
            # state = tf.keras.layers.Activation('sigmoid')(x)
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.LeakyReLU()(x)

        state = x
        # state = self.normalize_representation(x)
        x = tf.keras.layers.Conv2D(self.num_channels, kernel_size=(1, 1),
                                   kernel_initializer=self.init,data_format='channels_first')(x)
        x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dropout(0.05)(x, training=self.training)

        x = tf.keras.layers.Dense(int(self.num_channels/2), kernel_initializer=self.init)(x)
        x = tf.keras.layers.ELU()(x)
        # x = tf.keras.layers.ReLU()(x)
        # reward = tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='glorot_normal')(x)
        reward = tf.keras.layers.Dense(2*self.support_size+1)(x)
        model = tf.keras.models.Model(inp, [state, reward])
        return model

    def prediction_symbolic(self, encoded_state, training):
        policy, value = self.prediction_model(encoded_state, training=training)
        return policy, value

    def prediction_symbolic_scalar(self, encoded_state, training):
        policy, value = self.prediction_symbolic(encoded_state, training=training)
        value = support_to_scalar(value, self.support_size)
        return policy, value

    def representation_symbolic(self, state_t, training):
        encoded_state = self.representation_model(state_t, training=training)
        encoded_state = self.normalize_representation(encoded_state)
        return encoded_state

    def dynamics_symbolic(self, encoded_state_t, action_t, training):
        import tensorflow as tf
        action_reshaped = tf.reshape(action_t, shape=(-1, 1, 3, 3))

        dynam_inp = tf.concat(values=[encoded_state_t, action_reshaped], axis=1)
        next_encoded_state, reward = self.dynamics_model(dynam_inp,training=training)
        next_encoded_state = self.normalize_representation(next_encoded_state)
        return next_encoded_state, reward

    def dynamics_symbolic_scalar(self,encoded_state_t,action_t,training):
        next_encoded_state, reward = self.dynamics_symbolic(encoded_state_t,action_t,training=training)
        reward = support_to_scalar(reward, self.support_size)
        return next_encoded_state, reward

    def initial_inference_symbolic(self,inp_t,training):

        encoded_state_t = self.representation_symbolic(inp_t,training=training)
        policy_logits_t, value_t = self.prediction_symbolic(encoded_state_t,training=training)
        return encoded_state_t, policy_logits_t, value_t

    def initial_inference_symbolic_scalar(self, inp_t, training):
        encoded_state_t = self.representation_symbolic(inp_t,training=training)
        policy_logits_t, value_t = self.prediction_symbolic_scalar(encoded_state_t,training=training)
        return encoded_state_t, policy_logits_t, value_t

    def recurrent_inference_symbolic(self,hidden_tensor,action_tensor,training):
        h_tensor,r_tensor = self.dynamics_symbolic(hidden_tensor,action_tensor,training=training)
        policy_t,value_t = self.prediction_symbolic(h_tensor,training=training)
        return h_tensor, r_tensor, policy_t, value_t

    def recurrent_inference_symbolic_scalar(self,hidden_tensor,action_tensor,training):
        h_tensor,r_tensor = self.dynamics_symbolic_scalar(hidden_tensor,action_tensor,training=training)
        policy_t,value_t = self.prediction_symbolic_scalar(h_tensor,training=training)
        return h_tensor, r_tensor, policy_t, value_t
    #
    # def initial_inference(self, image):
    #     # representation + prediction function
    #     # if len(image.shape)==len(self.input_shape):
    #     # image = np.expand_dims(image,axis = 0)
    #     hidden, policy, value = self.sess.run([self.hidden_t_scalar,self.policy_t_scalar,self.value_t_scalar],
    #                                         {self.input_ph:image})
    #
    #     return (value[0,0], 0, policy, hidden)

    # def initial_inference_batch(self, image):
    # # representation + prediction function
    #     # if len(image.shape)==len(self.input_shape):
    #     hidden, policy, value = self.sess.run([self.hidden_t_scalar, self.policy_t_scalar, self.value_t_scalar],
    #                                           {self.input_ph: image})
    #     return value,0,policy,hidden

    # def recurrent_inference(self, hidden_state, action):
    #     # dynamics + prediction function
    #     # hidden_state = np.expand_dims(hidden_state,axis = 0)
    #     # action_in = np.zeros([1,self.num_actions])
    #     # action_in[0,int(action)] = 1
    #     action_in = np.full((1,self.num_actions), float(action)/self.num_actions)
    #     next_hidden, next_reward, policy, value = self.sess.run([self.next_hidden_t_scalar, self.reward_t_scalar,
    #                                                              self.next_policy_t_scalar, self.next_value_t_scalar],
    #                                                             {self.hidden_ph: hidden_state,
    #                                                              self.action_ph: action_in})
    #     return value[0,0],next_reward[0,0],policy,next_hidden

    def get_representation(self,state_in):
        rep = self.sess.run(self.representation, {self.input_ph:state_in})
        rep = self.sess.run(self.representation_normed, {self.hidden_ph: rep})
        return rep

    def get_prediction(self,repr):
        policy, value = self.sess.run([self.policy, self.value], {self.hidden_ph: repr})
        return policy,value

    def support_to_scalar_run(self, supp_in):
        out = self.sess.run(self.scalar_out, {self.support_ph: supp_in})
        return out


    def initial_inference(self, image):
        # representation + prediction function
        # if len(image.shape)==len(self.input_shape):
        # image = np.expand_dims(image,axis = 0)

        hidden = self.get_representation(image)
        policy, value = self.get_prediction(hidden)
        value = self.support_to_scalar_run(value)
        return value[0, 0], 0, policy, hidden

    def concat_state_and_action(self, encoded_state_t, action_t):
        import tensorflow as tf
        action_reshaped = tf.reshape(action_t, shape=(-1, 1, 3, 3))
        dynam_inp = tf.concat(values=[encoded_state_t, action_reshaped], axis=1)
        return dynam_inp

    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function
        # hidden_state = np.expand_dims(hidden_state,axis = 0)
        # action_in = np.zeros([1,self.num_actions])
        # action_in[0,int(action)] = 1
        action_in = np.full((1,self.num_actions), float(action)/self.num_actions)
        next_state, next_reward = self.sess.run([self.next_state, self.next_reward],
                                                {self.hidden_ph: hidden_state,
                                                 self.action_ph: action_in})
        next_state = self.sess.run(self.representation_normed, {self.hidden_ph: next_state})
        next_policy, next_value = self.get_prediction(next_state)
        next_value = self.support_to_scalar_run(next_value)
        next_reward = self.support_to_scalar_run(next_reward)
        return next_value[0,0], next_reward[0,0], next_policy, next_state




    def get_weights(self):
        # self.variables.get_weights()
        return [self.representation_model.get_weights(),
                self.prediction_model.get_weights(),
                self.dynamics_model.get_weights()]

    def set_weights(self, weights_list):
        # self.variables.set_weights(weights)
        self.representation_model.set_weights(weights_list[0])
        self.prediction_model.set_weights(weights_list[1])
        self.dynamics_model.set_weights(weights_list[2])

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps_trained

    def set_training_steps(self,s) -> int:
        self.steps_trained = s
        return 0

    # def load_weights(self,path):
    #     with open(path, 'rb') as f:
    #         s, w = pickle.load(f)
    #     self.set_weights(w)
    #     self.set_training_steps(s)

def support_to_scalar(logits, support_size):
    import tensorflow as tf
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = tf.nn.softmax(logits, axis=-1)
    idxes = np.arange(-support_size,support_size+1)
    ids = tf.cast(tf.constant(idxes),tf.float32)
    ids= tf.expand_dims(ids,axis = 0)
    x = tf.reduce_sum(ids * probabilities, axis=-1, keepdims=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    _x = (tf.sqrt(1 + 4 * 0.001 * (tf.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)
    x = tf.sign(x) * (tf.square(_x)-1)
    return x
