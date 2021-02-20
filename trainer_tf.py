import copy
import time

import numpy as np
import ray


import models_tf as models
from models_tf import support_to_scalar

@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        import tensorflow as tf
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        self.model = models.MuZeroNetwork(self.config, True)

        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))

        self.training_step = initial_checkpoint["training_step"]
        if self.config.optimizer == "SGD":
            self.optimizer = tf.train.MomentumOptimizer(self.config.lr_init, self.config.momentum)
        elif self.config.optimizer == "Adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.config.lr_init)
            self.optimizer_value = tf.compat.v1.train.AdamOptimizer(self.config.lr_init)
            self.optimizer_reward = tf.compat.v1.train.AdamOptimizer(self.config.lr_init)
            self.optimizer_policy = tf.compat.v1.train.AdamOptimizer(self.config.lr_init)

            # self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.config.lr_init)
            # self.optimizer = AdamW(lr = self.config.lr_init, weight_decay = self.config.weight_decay)
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )
        self.support_size = self.model.support_size
        self.sess = self.model.sess
        self.image_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image_ph')
        self.actions_ph = tf.placeholder(tf.float32, shape=[None, self.config.num_unroll_steps + 1,
                                                            self.model.num_actions], name='actions_ph')
        self.target_value_ph = tf.placeholder(tf.float32, shape=[None, None, 1], name='value_ph')
        self.target_reward_ph = tf.placeholder(tf.float32, shape=[None, None, 1], name='reward_ph')
        self.target_policy_ph = tf.placeholder(tf.float32, shape=[None, None, self.model.num_actions],
                                               name='policy_ph')
        self.grad_scale_ph = tf.placeholder(tf.float32, shape=[None, None], name='grad_scale_ph')
        self.PER_weights_ph = tf.placeholder(tf.float32, [None], name='PER_weights')

        [self.priorities,
         self.total_loss,
         self.value_loss,
         self.reward_loss,
         self.policy_loss,
         self.make_train_step,
         # self.make_train_step_value,
         # self.make_train_step_reward,
         # self.make_train_step_policy
         ] = self.train_symbolic(self.image_ph, self.actions_ph, self.target_value_ph,
                                                     self.target_reward_ph, self.target_policy_ph, self.PER_weights_ph,
                                                     self.grad_scale_ph)
        tmp_w = self.model.get_weights()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.model.set_weights(tmp_w)


    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            # self.update_lr()
            image, actions, target_value, target_reward, target_policy, per_weights, grad_scale = batch
            image = np.asarray(image)
            actions = np.asarray(actions)
            target_value = np.asarray(target_value)
            target_reward = np.asarray(target_reward)
            target_policy = np.asarray(target_policy)
            per_weights = np.asarray(per_weights)
            grad_scale = np.asarray(grad_scale)

            inp_img = image#np.transpose(image, [0, 2, 3, 1])
            actions_ohe = np.array(actions).astype(float)/self.model.num_actions
            actions_ohe = np.expand_dims(actions_ohe,axis = -1)
            actions_ohe = np.repeat(actions_ohe, self.model.num_actions,axis=-1)
            if not self.config.PER:
                per_weights = np.ones(image.shape[0])
            inp_dict = {self.image_ph:inp_img,
                        self.actions_ph:actions_ohe,
                        self.target_value_ph: np.expand_dims(target_value,axis = -1),
                        self.target_reward_ph: np.expand_dims(target_reward,axis = -1),
                        self.target_policy_ph:target_policy,
                        self.PER_weights_ph:per_weights,
                        self.grad_scale_ph:grad_scale}
            # _ = self.sess.run([self.make_train_step_value], inp_dict)
            # _ = self.sess.run([self.make_train_step_reward], inp_dict)
            # _ = self.sess.run([self.make_train_step_policy], inp_dict)
            # priorities, total_loss, value_loss, reward_loss, policy_loss = self.sess.run([self.priorities,
            #                                                                                  self.total_loss,
            #                                                                                  self.value_loss,
            #                                                                                  self.reward_loss,
            #                                                                                  self.policy_loss],
            #                                                                                 inp_dict)
            priorities, total_loss, value_loss, reward_loss, policy_loss, _ = self.sess.run([self.priorities,
                                                                                                    self.total_loss,
                                                                                                    self.value_loss,
                                                                                                    self.reward_loss,
                                                                                                    self.policy_loss,
                                                                                                    self.make_train_step],
                                                                                                    inp_dict)
            self.training_step += 1

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            info_to_storage = ray.put({
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": None#copy.deepcopy(models.dict_to_cpu(self.optimizer.state_dict())),
                    })
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(info_to_storage)
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()

            info_to_storage = ray.put({
                    "training_step": self.training_step,
                    "lr": self.config.lr_init,#self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                })
            shared_storage.set_info.remote(info_to_storage)

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)


    # def train_symbolic(self, image,actions, target_value,target_reward,target_policy,per_weights,grad_scale):
    #     import tensorflow as tf
    #     # Initial step, from the real observation.
    #
    #     # hidden_state, policy_logits, value = self.model.initial_inference_symbolic(image, training=True)
    #     hidden_state = self.model.representation_model(image, training=True)
    #     hidden_state = self.model.normalize_representation(hidden_state)
    #     policy_logits, value = self.model.prediction_model(hidden_state, training=True)
    #     # predictions = [(1.0, value, reward, policy_logits)]
    #     pred_values = [value]
    #     pred_rewards = []
    #     pred_policy_logits = [policy_logits]
    #
    #     # Recurrent steps, from action and previous hidden state.
    #
    #     for i in range(1, self.config.num_unroll_steps+1):
    #         action_reshaped = tf.reshape(actions[:, i, :], shape=(-1, 1, 3, 3))
    #         dynam_inp = tf.concat(values=[hidden_state, action_reshaped], axis=1)
    #         hidden_state, reward = self.model.dynamics_model(dynam_inp, training=True)
    #         hidden_state = self.model.normalize_representation(hidden_state)
    #         policy_logits, value = self.model.prediction_model(hidden_state, training=True)
    #         # hidden_state, reward, policy_logits, value = self.model.recurrent_inference_symbolic(
    #         #                                                     hidden_state,
    #         #                                                     actions[:, i, :], training=True)
    #         hidden_state = self.scale_gradient(hidden_state, 0.5)
    #
    #         pred_values.append(value)
    #         pred_rewards.append(reward)
    #         pred_policy_logits.append(policy_logits)
    #
    #
    #     target_value_scalar = target_value
    #     target_value = scalar_to_support(target_value, self.support_size)
    #     target_reward = scalar_to_support(target_reward, self.support_size)
    #
    #     pred_values = tf.stack(pred_values, axis=1)
    #     pred_rewards = tf.stack(pred_rewards, axis=1)
    #     pred_policy_logits = tf.stack(pred_policy_logits, axis=1)
    #
    #     priority = tf.pow(tf.abs(support_to_scalar(pred_values, self.support_size) - target_value_scalar),
    #                       self.config.PER_alpha)
    #
    #     grad_scale_cur = 1. / grad_scale
    #
    #     value_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_values, labels=target_value)
    #     value_loss_0 = value_loss[:, :1]
    #     value_loss = self.scale_gradient(value_loss[:, 1:], grad_scale_cur[:, 1:])
    #     value_loss = tf.concat(values=[value_loss_0, value_loss], axis=1)
    #     value_loss = tf.reduce_sum(value_loss, axis=1)
    #
    #     policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_policy_logits, labels=target_policy)
    #     policy_loss_0 = policy_loss[:, :1]
    #     policy_loss = self.scale_gradient(policy_loss[:, 1:], grad_scale_cur[:, 1:])
    #     policy_loss = tf.concat(values=[policy_loss_0, policy_loss], axis=1)
    #     policy_loss = tf.reduce_sum(policy_loss, axis=1)
    #
    #     reward_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_rewards, labels=target_reward[:, 1:, :])
    #     reward_loss = self.scale_gradient(reward_loss, grad_scale_cur[:, 1:])
    #     reward_loss = tf.reduce_sum(reward_loss, axis=1)
    #
    #     loss = self.config.value_loss_weight * value_loss + reward_loss + policy_loss
    #
    #     if self.config.PER:
    #         # Correct PER bias by using importance-sampling (IS) weights
    #         loss *= per_weights
    #
    #     loss = tf.reduce_mean(loss)
    #
    #     loss_value = tf.reduce_mean(self.config.value_loss_weight * value_loss)
    #     loss_reward = tf.reduce_mean(reward_loss)
    #     loss_policy = tf.reduce_mean(policy_loss)
    #
    #     params = (self.model.representation_model.trainable_weights +
    #               self.model.prediction_model.trainable_weights +
    #               self.model.dynamics_model.trainable_weights)
    #
    #     gvs = self.optimizer_value.compute_gradients(loss_value)
    #     reg_gvs = [(grad + self.config.weight_decay * var, var) for grad, var in gvs if
    #                ((var in params)  # and ('bias' not in var.name)
    #                                  )]        # capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
    #     step_value = self.optimizer_value.apply_gradients(reg_gvs)
    #
    #     gvs = self.optimizer_value.compute_gradients(loss_reward)
    #     reg_gvs = [(grad + self.config.weight_decay * var, var) for grad, var in gvs if
    #                ((var in params)  # and ('bias' not in var.name)
    #                                  )]        # capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
    #     step_reward = self.optimizer_reward.apply_gradients(reg_gvs)
    #
    #     gvs = self.optimizer_value.compute_gradients(loss_policy)
    #     reg_gvs = [(grad + self.config.weight_decay * var, var) for grad, var in gvs if
    #                ((var in params)# and ('bias' not in var.name)
    #                 )]
    #     # capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
    #     step_policy = self.optimizer_policy.apply_gradients(reg_gvs)
    #
    #     # for i, weights in enumerate(tf.trainable_variables()):
    #     #     # if 'bias' not in weights.name:
    #     #     if i ==0:
    #     #         w_loss = tf.nn.l2_loss(weights)
    #     #     else:
    #     #         w_loss += tf.nn.l2_loss(weights)
    #         # vars.append(weights)
    #     #
    #     # loss_reg = loss + self.config.weight_decay * w_loss
    #
    #     params = (self.model.representation_model.trainable_weights +
    #               self.model.prediction_model.trainable_weights +
    #               self.model.dynamics_model.trainable_weights)
    #
    #     gvs = self.optimizer.compute_gradients(loss)
    #     reg_gvs = [(grad + self.config.weight_decay * var, var) for grad, var in gvs if ((var in params)
    #                                                                                      # and ('bias' not in var.name)
    #                                                                                      )]
    #     # capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
    #     train_step = self.optimizer.apply_gradients(reg_gvs)
    #
    #
    #     # train_step = self.optimizer.minimize(loss_reg)
    #     # train_step = self.optimizer.minimize(loss, tf.trainable_variables())
    #
    #
    #     return [priority, tf.reduce_mean(loss), tf.reduce_mean(value_loss), tf.reduce_mean(reward_loss), tf.reduce_mean(policy_loss),
    #             train_step, step_value, step_reward, step_policy]


    def train_symbolic(self, image,actions, target_value,target_reward,target_policy,per_weights,grad_scale):
        import tensorflow as tf
        # Initial step, from the real observation.

        # hidden_state, policy_logits, value = self.model.initial_inference_symbolic(image, training=True)
        hidden_state = self.model.representation_model(image, training=True)
        hidden_state = self.model.normalize_representation(hidden_state)
        policy_logits, value = self.model.prediction_model(hidden_state, training=True)
        # predictions = [(1.0, value, reward, policy_logits)]
        pred_values = [value]
        pred_rewards = []
        pred_policy_logits = [policy_logits]

        # Recurrent steps, from action and previous hidden state.

        for i in range(1, self.config.num_unroll_steps+1):
            action_reshaped = tf.reshape(actions[:, i, :], shape=(-1, 1, 3, 3))
            dynam_inp = tf.concat(values=[hidden_state, action_reshaped], axis=1)
            hidden_state, reward = self.model.dynamics_model(dynam_inp, training=True)
            hidden_state = self.model.normalize_representation(hidden_state)
            policy_logits, value = self.model.prediction_model(hidden_state, training=True)
            # hidden_state, reward, policy_logits, value = self.model.recurrent_inference_symbolic(
            #                                                     hidden_state,
            #                                                     actions[:, i, :], training=True)
            hidden_state = self.scale_gradient(hidden_state, 0.5)

            pred_values.append(value)
            pred_rewards.append(reward)
            pred_policy_logits.append(policy_logits)


        target_value_scalar = target_value
        target_value = scalar_to_support(target_value, self.support_size)
        target_reward = scalar_to_support(target_reward, self.support_size)

        value, policy_logits = pred_values[0],pred_policy_logits[0]
        # Ignore reward loss for the first batch step
        current_value_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=value, labels=target_value[:,0,:])
        current_policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_logits, labels=target_policy[:,0,:])

        value_loss = current_value_loss
        policy_loss = current_policy_loss

        priorities = []
        # Compute priorities for the prioritized replay (See paper appendix Training)
        priority = tf.pow(tf.abs(support_to_scalar(value, self.support_size) - target_value_scalar[:, 0, :]),
                          self.config.PER_alpha)
        priorities.append(priority)


        for i in range(1, self.config.num_unroll_steps+1):
            value, reward, policy_logits = pred_values[i], pred_rewards[i-1], pred_policy_logits[i]

            current_value_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=value, labels=target_value[:, i, :])
            current_reward_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=reward, labels=target_reward[:, i, :])
            current_policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_logits,
                                                                          labels=target_policy[:, i, :])
            # Scale gradient by the number of unroll steps (See paper appendix Training)

            grad_scale_cur = 1./grad_scale[:,i]
            current_value_loss = self.scale_gradient(current_value_loss, grad_scale_cur)
            current_reward_loss = self.scale_gradient(current_reward_loss, grad_scale_cur)
            current_policy_loss = self.scale_gradient(current_policy_loss, grad_scale_cur)

            value_loss += current_value_loss
            if i ==1:
                reward_loss = current_reward_loss
            else:
                reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            priority = tf.pow(tf.abs(support_to_scalar(value, self.support_size) - target_value_scalar[:,i,:]),
                              self.config.PER_alpha)
            priorities.append(priority)

        priority = tf.stack(priorities,axis = 1)
        loss = self.config.value_loss_weight*value_loss+reward_loss+policy_loss


        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= per_weights

        loss = tf.reduce_mean(loss)

        # vars = []



        # for i, weights in enumerate(tf.trainable_variables()):
        #     # if 'bias' not in weights.name:
        #     if i ==0:
        #         w_loss = tf.nn.l2_loss(weights)
        #     else:
        #         w_loss += tf.nn.l2_loss(weights)
            # vars.append(weights)
        #
        # loss_reg = loss + self.config.weight_decay * w_loss

        params = (self.model.representation_model.trainable_weights +
                  self.model.prediction_model.trainable_weights +
                  self.model.dynamics_model.trainable_weights)

        gvs = self.optimizer.compute_gradients(loss)
        reg_gvs = [(grad + self.config.weight_decay * var, var) for grad, var in gvs if ((var in params)
                                                                                         and ('bias' not in var.name)
                                                                                         )]
        # capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
        train_step = self.optimizer.apply_gradients(reg_gvs)


        # train_step = self.optimizer.minimize(loss_reg)
        # train_step = self.optimizer.minimize(loss, tf.trainable_variables())


        return [priority, tf.reduce_mean(loss), tf.reduce_mean(value_loss), tf.reduce_mean(reward_loss), tf.reduce_mean(policy_loss), train_step]

    def scale_gradient(self,x,scale):
        import tensorflow as tf
        """Scales the gradient for the backward pass."""
        return x * scale + tf.stop_gradient(x) * tf.stop_gradient(1 - scale)

def loss_function_2(x_pred,target_x):
    import tensorflow as tf
    # Cross-entropy seems to have a better convergence than MSE
    loss = tf.reduce_sum((-target_x * tf.nn.log_softmax(x_pred,axis= -1)),axis = -1)
    # reward_pred = K.clip(reward_pred, 1e-5, 1-(1e-5))
    # loss = tf.reduce_sum(-(target_reward*tf.log(reward_pred)+(1.-target_reward)*tf.log(1.-reward_pred)),
    #                      axis=-1)
    return loss

def scalar_to_support(x, support_size):
    import tensorflow as tf
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    support_size = int(support_size)
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + 0.001*x

    # Encode on a vector
    x = tf.keras.backend.clip(x, -support_size, support_size)
    floor = tf.floor(x)
    prob = x - floor
    shape = tf.shape(x)
    logits = tf.zeros((shape[0], shape[1], 2 * support_size + 1))

    idxes = tf.constant(np.arange(support_size*2+1,dtype = np.int32))

    ids = tf.reshape(tf.cast(idxes,tf.int32),(1,1,-1))

    ids_to_fill = (floor + support_size)
    fill_mask = tf.cast(tf.equal(ids,tf.cast(ids_to_fill,tf.int32)),tf.float32)
    logits = (1 - prob) * fill_mask + logits * (1 - fill_mask)


    ids_to_fill = (floor + support_size + 1)
    mask = tf.cast(tf.less(2 * support_size, tf.cast(ids_to_fill,tf.int32)),tf.float32)
    prob = prob*(1-mask)
    ids_to_fill = ids_to_fill*(1-mask)
    fill_mask = tf.cast(tf.equal(ids, tf.cast(ids_to_fill, tf.int32)), tf.float32)
    logits = prob * fill_mask + logits * (1 - fill_mask)

    return logits


# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import state_ops
# from tensorflow.python.framework import ops
# from tensorflow.python.training import optimizer
#
#
# class AdamW(optimizer.Optimizer):
#
#     def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
#                  epsilon=1e-8, decay=0., weight_decay=0.025,
#                  batch_size=1, samples_per_epoch=1,
#                  epochs=1, use_locking=False, name="adamw"):
#         super(AdamW, self).__init__(use_locking, name)
#         self.lr = lr
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.decay = decay
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         self.epsilon = epsilon
#         self.samples_per_epoch = samples_per_epoch
#         self.epochs = epochs
#         self.initial_decay = decay
#
#         # Tensor versions of the constructor arguments, created in _prepare().
#         self._lr_t = None
#         self._beta_1_t = None
#         self._beta_2_t = None
#         self._decay_t = None
#         self._weight_decay_t = None
#         self._batch_size_t = None
#         self._epsilon_t = None
#         self._samples_per_epoch_t = None
#         self._epochs_t = None
#         self._initial_decay_t = None
#
#         self.iterations = 0
#
#     def _prepare(self):
#         self._lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")
#         self._beta_1_t = ops.convert_to_tensor(self.beta_1, name="_beta_1_t")
#         self._beta_2_t = ops.convert_to_tensor(self.beta_2, name="_beta_1_t")
#         # self._decay_t = ops.convert_to_tensor(self._decay, name="_beta_1_t")
#         self._weight_decay_t = ops.convert_to_tensor(self.weight_decay, name="_beta_1_t")
#         self._epsilon_t = ops.convert_to_tensor(self.epsilon, name="_beta_1_t")
#         # self._samples_per_epoch_t = ops.convert_to_tensor(self.samples_per_epoch, name="_beta_1_t")
#         # self._epochs_t = ops.convert_to_tensor(self.epochs, name="_beta_1_t")
#         # self._initial_decay_t = ops.convert_to_tensor(self._initial_decay, name="_beta_1_t")
#
#
#     def _create_slots(self, var_list):
#         # Create slots for the first and second moments.
#         for v in var_list:
#             self._zeros_slot(v, "vs", self._name)
#             self._zeros_slot(v, "ms", self._name)
#
#     def _resource_apply_dense(self, grad, var):
#         # self.iterations = 0#+=1
#         # t =self.iterations + 1
#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (1. / (1. + self.decay * self.iterations))
#         t = self.iterations+1
#         lr_t = lr * (tf.keras.backend.sqrt(1. - tf.keras.backend.pow(self.beta_2, t)) /
#                      (1. - tf.keras.backend.pow(self.beta_1, t)))
#
#         # lr_t = math_ops.cast(lr, var.dtype.base_dtype)
#
#
#
#         eps = 1e-7  # cap for moving average
#
#         m = self.get_slot(var, "ms")
#         v = self.get_slot(var, "vs")
#
#         grad = grad + self.weight_decay * var
#         m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad
#         v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.keras.backend.square(grad)
#
#         tf.keras.backend.update(m, m_t)
#         tf.keras.backend.update(v, v_t)
#
#         eta_t = 1.
#
#         upd = eta_t * (lr_t * m_t / (tf.keras.backend.sqrt(v_t) + self.epsilon))
#         # p_t = var -
#         # w_d = self.weight_decay * tf.keras.backend.sqrt(self.batch_size_t / (self.samples_per_epoch_t * self.epochs))
#         w_d = self.weight_decay# * np.sqrt(self.batch_size / (self.samples_per_epoch * self.epochs))
#         # p_t = p_t - eta_t * (w_d * var)
#         # if 'bias' not in var.name:
#         # upd = upd + eta_t * w_d * var
#
#
#
#         var_update = state_ops.assign_sub(var, upd)
#         # Create an op that groups multiple operations
#         # When this op finishes, all ops in input have finished
#         return var_update#control_flow_ops.group(*[var_update, m_t, v_t])
#
#
#     def _apply_sparse(self, grad, var):
#         raise NotImplementedError("Sparse gradient updates are not supported.")