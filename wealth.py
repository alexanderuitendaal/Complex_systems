import numpy as np
import matplotlib.pyplot as plt

# global variables
total_pop = 1000
beta_array = np.linspace(0.1,3,30)
market_growth_rate = 0.02
delta_omega = 0.01
time_period = 50


def plot_lorenz(pop, color):

    pop_2 = pop.copy()
    pop_2.sort()
    final_wealth = sum(pop_2)

    cumulative_wealth = [] # cumulative wealth as a percentage of total wealth

    for i in range(101):
        cumulative_wealth.append(((sum(pop_2[:int(i/100 * len(pop))]))/final_wealth)*100)

    x = np.linspace(0,100,101)

    plt.plot(x, cumulative_wealth, color = color)
    plt.plot(x,x, color = 'b')

    plt.text(2, 7, 'equality line',
             rotation=45,
             horizontalalignment='center',
             verticalalignment='center',
             multialignment='center')


    plt.xlabel(' % percentile of population')
    plt.ylabel('cumulatative wealth as a percentage')
    plt.title('lorenz curve')

    plt.show()

def plot_wealth(pop, color):

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

    plt.scatter(20, b20, c = color)
    plt.scatter(40,b40, c = color)
    plt.scatter(60, b60, c = color)
    plt.scatter(80,b80, c = color)
    plt.scatter(95, b95, c = color)
    plt.plot([20,40,60,80, 95], [b20, b40, b60, b80, b95], color = color)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('percentile')
    plt.ylabel('log_wealth')


    plt.show()


def initial_distribution(type):

    # TODO : initialisation of other distributions such as exponential or pareto
    # TODO : make realistic input distribition based on real data

    if type == 'exponential':
        return np.random.exponential(50, 10)

    if type == 'powerlaw':
        return np.random.pareto(.5)

    if type == 'pareto':
        return np.random.pareto(.5)

    if type == 'uniform':
        return np.random.uniform(25,75)


def calc_gini(population):


    pop_gini = population
    sorted_pop = sorted(pop_gini)
    n = len(sorted_pop)
    temp_sum = 0.0

    for i in range(n):
        temp_sum += i * sorted_pop[i]

    temp_sum *= 2

    gini_coef = (temp_sum/ (n * sum(sorted_pop))) - ((n + 1) / n)
    return gini_coef

# wealth distribution

def wealth_power(wealth, beta):
    return np.power(wealth, beta)



# create population
pop = np.empty(total_pop, dtype=object)
for i in range(len(pop)):
    pop[i] = initial_distribution('uniform')


results_dict = {}
gini_list = []


for beta in beta_array:
    print('new round')


# calulate iterations
    max_iter = np.sum(pop) * (np.exp(market_growth_rate* time_period ) -1)/delta_omega

    # plot begin wealth

    # plot_wealth(pop, color = 'b')
    # plot_lorenz(pop, color = 'r')

    # simulation
    chances = wealth_power(pop, beta)
    check = int(max_iter/10)
    sum_chances = sum(chances)

    for iter in range(int(max_iter)):

        if iter % check == 0:
            print(10*iter/max_iter)

        decision = np.random.uniform(0,sum_chances)
        decision_pos, chance_sum = 0, chances[0]

        # print(decision, sum(chances))

        while decision > chance_sum and decision_pos < total_pop - 1:
            decision_pos += 1
            chance_sum += chances[decision_pos]

        pop[decision_pos] += delta_omega
        new_wealth = wealth_power(pop[decision_pos], beta)
        sum_chances += new_wealth - chances[decision_pos]
        chances[decision_pos] = new_wealth


# plot_wealth(pop, color = 'b')
# plot_lorenz(pop, color = 'r')

# calculate gini coefficient and make an array, gini = 0 is perfect equality, gini == 1 is perfect inequality
#     gini_coef_ = np.empty(1)
#     gini_coef_[0] = calc_gini(pop)


    results_dict[beta] = pop
    gini_list.append(calc_gini(pop))

df = pd.DataFrame(results_dict)
print(gini_list)
df.to_pickle("./results_test_1.pkl")

# dict = {}
# for i in range(0,10):
#     array = np.random.randint(1,101,5)
#     dict[i] = array

# df = pd.DataFrame(dict)
#
# col_1 = df.iloc[:, 4]
# df.to_pickle("./testpickle_2.pkl")
# lala = pd.read_pickle("./testpickle_2.pkl")
#
# print(lala)


# save results of the simulation







# for the model, fix time market growth and pop to reasonable amounts (do tests), does the model work?
# run for 10 beta's, 10 lorenz curves and 10 gini's (enough to see a phase transition)

# hoelang duurt voor t = 50 met 10000 pop, 17:56 start,18.02 (20%), 18.05(30)% ca 30 minuten per run, dus als je snachts laat runnen ong 20 runs possible


# next, plot the final gini coeffecient against the beta, and maybe plot a few lorenz curves (the 4 most extremes)



