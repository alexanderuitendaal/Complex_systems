import numpy as np
import matplotlib.pyplot as plt

# global variables 
total_pop = 1000
beta = 1
omega = 1
market_growth_rate = 0.02588
delta_omega = 0.01
time_period = 5


def plot_wealth(pop):

    pop_2 = pop.copy()
    pop_2.sort()
    total_sum = max(pop_2)
    # ind_list = np.empty()

    b20 = sum(pop_2[:int(.2 * len(pop))])
    b40 = sum(pop_2[:int(.4 * len(pop))])
    b60 = sum(pop_2[:int(.6 * len(pop))])
    b80 = sum(pop_2[:int(.8 * len(pop))])
    b95= sum(pop_2[:int(.95 * len(pop))])

    # for percentile in np.linspace(0,1,100):
    #     ind = len(np.where(pop_2 < percentile * total_sum))
    #     np.append(ind_list, ind)

    plt.scatter(20, b20)
    plt.scatter(40,b40)
    plt.scatter(60, b60)
    plt.scatter(80,b80)
    plt.scatter(95, b95)

    plt.yscale('log')
    plt.xlabel('percentile')
    plt.ylabel('log_wealth')
    plt.show()




def initial_distribution(type):

    # TODO : initialisation of other distributions such as exponential or pareto

    # if type == 'exponential':

    # if type == 'pareto':

    # if type == 'powerlaw':

    if type == 'uniform':

        return np.random.uniform(0,100)



# class of person

# class Person:
#     def __init__(self, id):
#         self.id = id
#         self.wealth = initial_distribution('uniform')


# wealth distribution
def wealth_power(wealth, beta):
    return np.power(wealth, beta)



# create population
pop = np.empty(total_pop, dtype=object)
for i in range(len(pop)):
    pop[i] = initial_distribution('uniform')


# calulate iterations
max_iter = np.sum(pop) *(np.exp(market_growth_rate*time_period ) -1 )/delta_omega


# simulation
for iter in range(int(max_iter)):

    # if iter%10000 == 0:
    #     #     print(iter)
    chances =  wealth_power(pop, beta)


    decision = np.random.uniform(0,sum(chances))
    decision_pos, chance_sum = 0, pop[0]

    # print(decision, sum(chances))

    while decision > chance_sum and decision_pos < total_pop - 1:
        decision_pos += 1
        chance_sum += chances[decision_pos]

    pop[decision_pos] += delta_omega


plot_wealth(pop)




