from extern_lib import *


def state2tensor(states):
    inputs = np.zeros([max_input_length, 2])
    i = 0
    for price in states:
        inputs[i][0] = price[0]
        inputs[i][1] = price[1]
        i += 1
    inputs = tf.constant(inputs, dtype=tf.float32)
    inputs = tf.expand_dims(inputs, axis=0)
    return inputs


class ReplayBuffer():
    # 经验回放池
    def __init__(self):
        # 双向队列
        self.buffer = collections.deque(maxlen=50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self,n):
        # 从回放池采样n个5元组
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        # 按类别进行整理
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(state2tensor(s))
            a_lst.append(tf.expand_dims(tf.constant(a, dtype=tf.float32),axis=0))
            r_lst.append(r)
            s_prime_lst.append(state2tensor(s_prime))
            done_mask_lst.append(done_mask)
        # 转换成Tensor
        return tf.concat(s_lst, axis=0),\
            tf.concat(a_lst, axis=0), \
            tf.constant(r_lst, dtype=tf.float32), \
            tf.concat(s_prime_lst, axis=0), \
            tf.constant(done_mask_lst, dtype=tf.float32)

    def size(self):
        return len(self.buffer)
