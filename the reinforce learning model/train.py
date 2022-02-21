
from extern_lib import *
from agent import *
from ReplayBuffer import *
from test import *

if __name__ == '__main__':
    age = Agent()
    env = invest_game(datetime.datetime.strptime("9/12/16", "%m/%d/%y"),
                      datetime.datetime.strptime("2/20/17", "%m/%d/%y"))
    memory = ReplayBuffer()
    print_interval = 5
    optimizer = optimizers.Adam(lr=learning_rate)
    # all_actor_loss=[]
    # all_critic_loss=[]
    for n_epi in range(100):  # 训练次数
        # epsilon概率也会8%到1%衰减，越到后面越使用Q值最大的动作
        epsilon = max(0.01, 0.5 - 0.01 * n_epi )
        age.sample(env, memory,epsilon)
        if n_epi % print_interval == 0:
            age.shadow_update()
            age.save_model()
            print("test score:",test(age,env,epoch=n_epi),"memory size:",memory.size())
            sys.stdout.flush()
        if memory.size() > 2000:  # 缓冲池只有大于2000就可以训练
            for _ in range(10):
                actor_loss,critic_loss=age.train(memory)
            print("epoch:",n_epi,"actor_loss:",actor_loss,"critic_loss:",critic_loss)


