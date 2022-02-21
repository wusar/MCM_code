from cProfile import label
from os import stat
from agent import *
from env import *

def test_plot(dollar, gold, bitcoin, gold_price, bitcoin_price,plot_num=0):
    fig = plt.figure(plot_num)  # 新图 0
    # print(dollar,gold,bitcoin,gold_price,bitcoin_price)
    plt.plot(dollar,label="dollar")
    plt.plot(gold,label="gold")
    plt.plot(bitcoin,label="bitcoin")
    plt.plot(gold_price,label="gold_price")
    plt.plot(bitcoin_price,label="bitcoin_price")
    # 显示绘图结果
    plt.legend()
    plt.savefig("./figure/"+str(plot_num)+".png")
    # plt.show()
    plt. close(plot_num)  # 关闭图 0


def test(agent, env,epoch=0):
    env.reset()
    state, reward, done = env.step(0, 0)
    old_state = state
    dollar, gold, bitcoin, gold_price, bitcoin_price = [], [], [], [], []
    for _ in range(500):
        action = agent.action(state)[0]
        bought_gold, bought_bitcoin = float(action[0]), float(action[1])
        # print(bought_gold,bought_bitcoin)
        state, reward, done = env.step(bought_gold, bought_bitcoin)
        dollar.append(env.property[0])
        gold.append(env.property[1]*state[-1][0])
        bitcoin.append(env.property[2]*state[-1][1])
        gold_price.append(state[-1][0])
        bitcoin_price.append(state[-1][1])
        if done == True:
            break
    test_plot(dollar, gold, bitcoin, gold_price, bitcoin_price,epoch)
    return env.old_property


if __name__ == '__main__':
    age = Agent()
    env = invest_game(datetime.datetime.strptime("9/1/18", "%m/%d/%y"),
                      datetime.datetime.strptime("9/6/18", "%m/%d/%y"))
    print(test(age, env))
