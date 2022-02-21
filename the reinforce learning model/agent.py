from msilib import sequence
from extern_lib import *
from env import *
from ReplayBuffer import *


class actor(keras.Model):
    def __init__(self):
        super(actor, self).__init__()
        self.rnn1 = layers.SimpleRNN(units, dropout=0.2, return_sequences=True)
        self.rnn2 = layers.SimpleRNN(units, dropout=0.2)
        self.outlayer = Sequential([
            layers.Dense(32),
            layers.Dense(2)])

    def call(self, inputs, training=None):
        x = inputs
        sequence = self.rnn1(x)
        x = self.rnn2(sequence)
        x = self.outlayer(x)
        # x=tf.clip_by_norm(x,0.1)
        return x


class critic(keras.Model):
    def __init__(self):
        super(critic, self).__init__()
        self.rnn1 = layers.SimpleRNN(units, dropout=0.2, return_sequences=True)
        self.rnn2 = layers.SimpleRNN(units, dropout=0.2)
        self.outlayer = Sequential([
            layers.Dense(32),
            layers.Dense(1)])

    def call(self, inputs, training=None):
        state, action = inputs
        sequence = self.rnn1(state)
        x = self.rnn2(sequence)
        x = tf.concat([x, action], axis=1)
        x = self.outlayer(x)
        return x


class Agent():
    def __init__(self):
        self.actor = actor()
        self.actor.build(input_shape=(batch_size, max_input_length, 2))
        # self.actor.summary()
        self.critic = critic()
        self.critic.build(
            input_shape=[(batch_size, max_input_length, 2), (batch_size, 2)])
        # self.critic.summary()

        self.actor_shadow = actor()
        self.actor_shadow.build(input_shape=(batch_size, max_input_length, 2))
        self.critic_shadow = critic()
        self.critic_shadow.build(
            input_shape=[(batch_size, max_input_length, 2), (batch_size, 2)])

        self.actor_optimizer = optimizers.Adam(lr=learning_rate)
        self.critic_optimizer = optimizers.Adam(lr=learning_rate)

    def save_model(self):
        self.actor.save_weights('./model/actor.h5')
        self.critic.save_weights('./model/critic.h5')

    def load_model(self):
        self.actor.load_weights('./model/actor.h5')
        self.actor_shadow.load_weights('./model/actor.h5')
        self.critic.load_weights('./model/critic.h5')
        self.critic_shadow.load_weights('./model/critic.h5')

    def action(self, states, epsilon=0):
        inputs = state2tensor(states)
        action = self.actor(inputs)
        action += tf.random.normal(action.shape,
                                   mean=0.0, stddev=epsilon, dtype=tf.float32)
        return action

    def sample(self, env, memory, epsilon=0):
        env.reset()
        state, reward, done = env.step(0, 0)
        old_state = state
        for _ in range(500):
            action = self.action(state, epsilon)[0]
            bought_gold, bought_bitcoin = float(action[0]), float(action[1])
            state, reward, done = env.step(bought_gold, bought_bitcoin)
            done_mask = 0.0 if done else 1.0  # 结束标志掩码
            memory.put((old_state, action, reward, state, done_mask))
            old_state = state
            if done == True:
                break

    def shadow_update(self):
        for src, dest in zip(self.actor.variables, self.actor_shadow.variables):
            dest.assign(src*update_coefficient+dest*(1-update_coefficient))
        for src, dest in zip(self.critic.variables, self.critic_shadow.variables):
            dest.assign(src*update_coefficient+dest*(1-update_coefficient))

    def train(self, memory):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        huber = losses.Huber()
        with tf.GradientTape() as tape:
            q = self.critic((s, self.actor(s)))
            actor_loss = -tf.reduce_mean(q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        with tf.GradientTape() as tape:
            q = self.critic((s, a))
            max_q_prime = self.critic_shadow(
                (s_prime, self.actor_shadow(s_prime)))
            target = r + gamma * max_q_prime * done_mask
            critic_loss = huber(q, target)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))
        return float(actor_loss),float(critic_loss)


if __name__ == '__main__':
    a = Agent()
    env = invest_game(datetime.datetime.strptime("9/12/16", "%m/%d/%y"),
                      datetime.datetime.strptime("9/20/16", "%m/%d/%y"))
    mem = ReplayBuffer()
    a.sample(env, mem)
    state = tf.zeros((batch_size, max_input_length, 2))
    action = tf.zeros((batch_size, 2))
    a.critic((state, action))
