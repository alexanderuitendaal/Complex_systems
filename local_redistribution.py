
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import datetime
simulations = 50

'''
##################
## PLOTFUNCTION ##
##################
'''
def plot_wealth_and_degree(all_degrees, all_wealths):
    points = 100
    checking_wealths = np.logspace(-1, 4,points)
    observed_wealths = np.zeros(points)
    len_wealth = len(all_wealths)

    for i, x in enumerate(checking_wealths):
        observed_wealths[i] = len(np.where(all_wealths > x)[0])/len_wealth

    plt.plot(checking_wealths,observed_wealths,'--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('wealth')
    plt.ylabel('probability')
    plt.show()

    checking_degrees = np.logspace(0, 3,points)
    observed_degrees = np.zeros(points)
    len_degrees = len(all_degrees)

    for i, x in enumerate(checking_degrees):
        observed_degrees[i] = len(np.where(all_degrees > x)[0])/len_degrees

    plt.plot(checking_degrees,observed_degrees,'--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('probability')
    plt.show()

'''
##################
## MAIN FUNCTION #
##################
'''
def simulation():
    '''
    PARAMETERS
    '''
    pop = nx.Graph()
    current_persons, wealths = np.array([]), np.array([])
    pop_size = 1000
    gamma = 0.5
    beta1, beta2 = 1, 10


    # DO FIRST ITERATION EXPLICIT
    pop.add_node(0, wealth=0, ability=np.random.beta(beta1, beta2))
    current_persons = np.append(current_persons, 0)
    # print(pop.nodes[0]['wealth'])
    wealths = np.append(wealths, pop.nodes[0]['wealth'])

    # RUN MODEL FOR GIVEN POPULATION SIZE
    for person in range(1, pop_size):
        if person % 1000 == 0:
            print("step", int(person/1000),"/10")
        # add node
        pop.add_node(person, wealth=0, ability=np.random.beta(beta1, beta2))
        current_persons = np.append(current_persons, person)
        wealths = np.append(wealths, pop.nodes[person]['wealth'])

        # determine to which node is linked
        if sum(wealths) > 0:
            linked = np.random.choice(current_persons, p = wealths/sum(wealths))
        else:
            linked = np.random.choice(current_persons)
        pop.add_edge(linked, person)

        # produce wealth
        for node in pop.nodes:
            wealth_produced = np.random.binomial(pop.degree[node], pop.nodes[node]['ability'])
            wealths[node] += wealth_produced
            pop.nodes[node]['wealth'] += wealth_produced

            # local redistribution
            if len(list(pop.neighbors(node))) > 0:

                portion = gamma/len(list(pop.neighbors(node)))
                wealths[node] -= gamma * wealth_produced
                pop.nodes[node]['wealth'] -= gamma * wealth_produced

                for neighbor in pop.neighbors(node):
                    neighbor = int(neighbor)
                    wealths[neighbor] += portion * wealth_produced
                    pop.nodes[neighbor]['wealth'] += portion * wealth_produced

    return pop, wealths

'''
##################
###### REST ######
##################
'''
all_wealths, all_degrees = np.array([]), np.array([])
for simu in range(simulations):
    print(simu)
    pop, wealths = simulation()

    all_wealths = np.append(all_wealths, wealths)
    all_degrees = np.append(all_degrees, np.array(pop.degree)[:,1])

plot_wealth_and_degree(all_degrees, all_wealths)
# np.save("wealths1", wealths)
# nx.write_edgelist(pop, "network1.edgelist")

