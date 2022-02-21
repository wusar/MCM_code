
from extern_lib import *
def load_price():  # load price history
    gold_price = {}
    bitcoin_price = {}
    with open("BCHAIN-MKPRU.csv") as bitcoin_price_file:
        lines = bitcoin_price_file.readlines()
        for line in lines[1:]:  # ignore the first line
            date, price = line.split(',')
            date = datetime.datetime.strptime(date, "%m/%d/%y")
            price = float(price)
            bitcoin_price[date] = price
    # there are many dirty data which don't have price information,I simply delete them from the origin data
    with open("LBMA-GOLD.csv") as gold_price_file:
        lines = gold_price_file.readlines()
        for line in lines[1:]:  # ignore the first line
            date, price = line.split(',')
            date = datetime.datetime.strptime(date, "%m/%d/%y")
            price = float(price)
            gold_price[date] = price

    return gold_price, bitcoin_price


class invest_game:
    def __init__(self, start_date, end_date):
        # The commission for each transaction (purchase or sale) costs α% of the amount traded
        self.alpha_gold = 0.01
        self.alpha_bitcoin = 0.02

        # portfolio consisting of cash, gold, and bitcoin [C, G, B] in U.S. dollars, troy ounces, and bitcoins, respectively.
        self.property = [1000, 0, 0]
        self.glod_price, self.bitcoin_price = load_price()
        self.start_date = start_date  # the start date of the invest game
        self.end_date = end_date
        self.current_date = self.start_date
        # the most recent price of the gold
        date=self.start_date
        while(date not in self.glod_price):
            date-=datetime.timedelta(days=1)
        self.old_gold_price = self.glod_price[date]   
        self.old_property=1000

        self.price_history=[]

    def reset(self):
        self.property = [1000, 0, 0]
        self.old_property=1000
        self.current_date = self.start_date
        date=self.start_date
        while(date not in self.glod_price):
            date-=datetime.timedelta(days=1)
        self.old_gold_price = self.glod_price[date]   
        gold_price = self.glod_price[date]
        return (self.property, gold_price, self.bitcoin_price[self.current_date])

    def step(self, bought_gold, bought_bitcoin):  # buy gold and bitcoin,as a step of the game
        if bought_gold+self.property[1]<0:
            bought_gold=-self.property[1]
        if bought_bitcoin+self.property[2]<0:
            bought_bitcoin=-self.property[2]        
        # frist calculate the cost price
        if self.current_date in self.glod_price.keys():
            gold_price = self.glod_price[self.current_date]
            self.old_gold_price=gold_price
        else:
            gold_price = self.old_gold_price  # If the price of gold is not in the list, it means that the price of gold is not open
            bought_gold = 0
        bitcoin_price = self.bitcoin_price[self.current_date]
        cost = bought_gold*gold_price+self.alpha_gold * \
            abs(bought_gold*gold_price)+bought_bitcoin * bitcoin_price + self.alpha_bitcoin*abs(
                bought_bitcoin * bitcoin_price)
        # In case of overspending, multiply both gold and bitcoin bought by a factor to ensure money is positive
        if cost > self.property[0]:
            factor = self.property[0]/cost
            bought_gold *= factor
            bought_bitcoin *= factor
            cost = self.property[0]

        self.property[0] -= cost
        self.property[1] += bought_gold
        self.property[2] += bought_bitcoin

        self.current_date += datetime.timedelta(days=1)
        new_property=self.property[0]+gold_price*self.property[1]+bitcoin_price*self.property[2]
        reward=new_property-self.old_property
        self.old_property=new_property
        done=False
        if self.current_date==self.end_date:
            done=True
        # if self.old_property<200:
        #     done=True
        self.price_history.append((gold_price,bitcoin_price))
        history_length=len(self.price_history)
        if history_length<max_input_length:
            for i in range(max_input_length-history_length):
                self.price_history.insert(0,copy.deepcopy(self.price_history[0]))
        if history_length>max_input_length:
            self.price_history=self.price_history[history_length-max_input_length:]
        return copy.deepcopy(self.price_history),reward,done


if __name__ == '__main__':
    env = invest_game(datetime.datetime.strptime("9/12/16", "%m/%d/%y"),
                      datetime.datetime.strptime("9/20/18", "%m/%d/%y"))
    env.reset()
    while True:
        bought_gold=random.random()-0.5
        bought_bitcoin=random.random()-0.5
        state,reward,done=env.step(bought_gold,bought_bitcoin)
        print(state,reward)
        if done==True:
            break

