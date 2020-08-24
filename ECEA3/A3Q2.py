import random
import copy
import math
import matplotlib.pyplot as plt

cities = [
    (1, 1150, 1760),
    (2, 630, 1660),
    (3, 40, 2090),
    (4, 750, 1100),
    (5, 750, 2030),
    (6, 1030, 2070),
    (7, 1650, 650),
    (8, 1490, 1630),
    (9, 790, 2260),
    (10, 710, 1310),
    (11, 840, 550),
    (12, 1170, 2300),
    (13, 970, 1340),
    (14, 510, 700),
    (15, 750, 900),
    (16, 1280, 1200),
    (17, 230, 590),
    (18, 460, 860),
    (19, 1040, 950),
    (20, 590, 1390),
    (21, 830, 1770),
    (22, 490, 500),
    (23, 1840, 1240),
    (24, 1260, 1500),
    (25, 1280, 790),
    (26, 490, 2130),
    (27, 1460, 1420),
    (28, 1260, 1910),
    (29, 360, 1980)
]

# alpha and beta are balance the local vs global
alpha = 1
beta = 1
init_pher_val = 1
q_0 = 0.35
evap_rate = 0.4
Q = 40
pheromone = [[init_pher_val for i in range(len(cities))] for i in range(len(cities))]

class Ant:
  def __init__(self):
    self.solution = []
    self.current_city = None
    self.has_online_update = False
  
  def add_city(self, city):
    if not city in self.solution:
      self.solution.append(city)
      return True
    else:
      return False

  def is_tour_complete(self):
    return len(self.solution) == len(cities)

  def select_city(self):
    q = random.random()
    if q < q_0:
      city = self.greedy_city()
    else:
      city = self.probablistic_city()

    return city
  
  def greedy_city(self):
    highest_pheromone = 0
    best_city = None
    for city in cities:
      city_index = city[0]
      if city in self.solution or city_index == self.solution[-1][0]:
        continue
      
      if pheromone[self.solution[-1][0] - 1][city[0] - 1] > highest_pheromone:
        highest_pheromone = pheromone[self.solution[-1][0] - 1][city[0] - 1]
        best_city = city
    
    return best_city
  
  def roulette_selection(self, weights):
    totals=[]
    for i in range(len(weights)):
      if i == 0:
        totals.append(weights[0])
      else:
        totals.append(weights[i] + totals[i-1])

    random_weight = random.random() * totals[-1]

    for index, weight in enumerate(totals):
      if random_weight < weight:
        return index
  
  def probablistic_city(self):
    city_index = self.solution[-1][0] - 1
    available_cities = []
    individual_probability = []
    total_probability = 0

    for i in range(len(pheromone[city_index])):
      if i != city_index and not cities[i] in self.solution:
        individual_probability.append((pheromone[city_index][i] ** alpha) / (self.euclid_distance(self.solution[-1], cities[i]) ** beta))
        available_cities.append(cities[i])
        total_probability += individual_probability[-1]
    
    for i in range(len(individual_probability)):
      individual_probability[i] = individual_probability[i] / total_probability

    return available_cities[self.roulette_selection(individual_probability)]
  
  def offline_update(self, best_solution):
    cost = self.calc_solution_cost(other_solution=best_solution)
    solution_array = self.solution
    delta_tau = Q / cost

    for i in range(1, len(solution_array)):
      city1, city2 = solution_array[i-1][0] - 1, solution_array[i][0] - 1

      pheromone[city1][city2] = pheromone[city1][city2] + delta_tau
      pheromone[city2][city1] = pheromone[city2][city1] + delta_tau

  def online_update_delayed(self):
    cost = self.calc_solution_cost()
    solution_array = self.solution
    delta_tau = Q / cost

    for i in range(1, len(solution_array)):
      city1, city2 = solution_array[i-1][0] - 1, solution_array[i][0] - 1

      pheromone[city1][city2] = pheromone[city1][city2] + delta_tau
      pheromone[city2][city1] = pheromone[city2][city1] + delta_tau

  def euclid_distance(self, city1, city2):
    return math.sqrt( (city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2 )

  def calc_solution_cost(self, other_solution=None):
    cost = 0
    solution_array = self.solution
    if other_solution != None:
      solution_array = other_solution

    for i in range(1, len(solution_array)):
      cost += self.euclid_distance(solution_array[i-1], solution_array[i])

    cost += self.euclid_distance(solution_array[-1], solution_array[0])
    return cost

def evaporate_pheromone():
  for i in range(len(pheromone)):
    for j in range(len(pheromone[i])):
      pheromone[i][j] = pheromone[i][j] * (1 - evap_rate)

def reset_pheromone():
  for i in range(len(pheromone)):
    for j in range(len(pheromone[i])):
      pheromone[i][j] = init_pher_val

def create_ants(num_ants):
  ants = [Ant() for i in range(num_ants)]
  for ant in ants:
    random_city_index = random.randint(0, len(cities) - 1)
    ant.add_city(cities[random_city_index])
  
  return ants

def ACO(num_ants=10, num_iters=150, do_online_update=True):
  ant_costs = []

  for i in range(num_iters):
    ants = create_ants(num_ants)
    iter_best = 10000000
    for ant in ants:
      while not ant.is_tour_complete():
        ant.add_city(ant.select_city())
      if do_online_update:
        ant.online_update_delayed()
      ant_cost = ant.calc_solution_cost()
      if ant_cost < iter_best:
        iter_best = ant_cost
    evaporate_pheromone()
    ant_costs.append(iter_best)
  t = [i for i in range(len(ant_costs))]
  reset_pheromone()
  return t, ant_costs

def part1():
  t, y = ACO()

  plt.plot(t, y, label="Shortest Ant Distance per Iteration")
  plt.xlabel('Iteration')
  plt.ylabel('TSP Distance')
  plt.title('Shortest Ant Distance per Iteration')
  plt.legend()
  plt.show()

def part2():
  global evap_rate
  evap_rate = 0.3
  t, y1 = ACO()
  evap_rate = 0.5
  t, y2 = ACO()
  evap_rate = 0.8
  t, y3 = ACO()

  plt.plot(t, y1, label="0.3 Pheromone Persistence")
  plt.plot(t, y2, label="0.5 Pheromone Persistence")
  plt.plot(t, y3, label="0.8 Pheromone Persistence")
  plt.xlabel('Iteration')
  plt.ylabel('TSP Distance')
  plt.title('Shortest Ant Distance per Iteration')
  plt.legend()
  plt.show()

def part3():
  global q_0
  q_0 = 0.80
  t, y1 = ACO()
  q_0 = 0.50
  t, y2 = ACO()
  q_0 = 0.10
  t, y3 = ACO()

  plt.plot(t, y1, label="0.8 State Transition Control")
  plt.plot(t, y2, label="0.5 State Transition Control")
  plt.plot(t, y3, label="0.1 State Transition Control")
  plt.xlabel('Iteration')
  plt.ylabel('TSP Distance')
  plt.title('Shortest Ant Distance per Iteration')
  plt.legend()
  plt.show()

def part4():
  t, y1 = ACO(num_ants=75)
  t, y2 = ACO(num_ants=5)

  plt.plot(t, y1, label="75 Ants")
  plt.plot(t, y2, label="25 Ants")
  plt.xlabel('Iteration')
  plt.ylabel('TSP Distance')
  plt.title('Shortest Ant Distance per Iteration')
  plt.legend()
  plt.show()

def part5():
  t, y1 = ACO(do_online_update=False)
  t, y2 = ACO()

  plt.plot(t, y1, label="Online Update OFF")
  plt.plot(t, y2, label="Online Update ON")
  plt.xlabel('Iteration')
  plt.ylabel('TSP Distance')
  plt.title('Shortest Ant Distance per Iteration')
  plt.legend()
  plt.show()


def main():
  part1()
  part2()
  part3()
  part4()
  part5()
    
if __name__ == '__main__':
    main()
