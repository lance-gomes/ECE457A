import random
import matplotlib.pyplot as plt
from copy import deepcopy
from math import log, ceil
from perffcn import perffcn

precision = 2
parameter_interval = [(2.00,18.00), (1.05, 9.42), (0.26, 2.37)]
num_bits = [11, 10, 8]
p_m = 0.25
p_c = 0.6

def num_2_bit(num, lower_bound, n_bits):
  bi_str = str(bin(int(((num - lower_bound)*10*10)))[2:])
  while len(bi_str) < n_bits:
    bi_str = "0" + bi_str
  return bi_str

def vals_2_bit(args):
  ret = [
    num_2_bit(args[0], parameter_interval[0][0], num_bits[0]),
    num_2_bit(args[1], parameter_interval[1][0], num_bits[1]),
    num_2_bit(args[2], parameter_interval[2][0], num_bits[2])
  ]

  res = ret[0] + ret[1] + ret[2]
  return res

def rand_chrom():
  ret = [
    round(random.uniform(parameter_interval[0][0], parameter_interval[0][1]),2),
    round(random.uniform(parameter_interval[1][0], parameter_interval[1][1]),2),
    round(random.uniform(parameter_interval[2][0], parameter_interval[2][1]),2)
  ]
  return vals_2_bit(ret)

def fitness(chrom):
  pos1 = num_bits[0]
  pos2 = num_bits[0] + num_bits[1]
  args = [
    round(int(chrom[0:pos1], 2) / 100 + parameter_interval[0][0], 2),
    round(int(chrom[pos1: pos2], 2) / 100 + parameter_interval[1][0], 2),
    round(int(chrom[pos2:], 2) / 100 + parameter_interval[2][0], 2)
  ]

  try:
    ise, rise_time, settling_time, overshoot = perffcn(args[0], args[1], args[2])
  except:
    return 1/1000
  return 1/ise

def initialize_population(num_pop):
  population = []
  for i in range(num_pop):
    population.append(rand_chrom())
  return population

def next_gen(population, initial_population=50):
  sorted_population = sorted_pop(population)
  mating_pool_size = 25 if initial_population == 50 else initial_population // 2
  new_mating_pool = mating_pool(sorted_population, mating_pool_size)
  seen_pairs = []
  next_gen = []

  while(len(next_gen) <= initial_population - 2):
    p1_index = random.randint(0, mating_pool_size - 1)
    p2_index = random.randint(0, mating_pool_size - 1)

    p1 = new_mating_pool[p1_index]
    p2 = new_mating_pool[p2_index]

    if p1 == p2 or (p1, p2) in seen_pairs or (p2, p1) in seen_pairs:
      continue
    
    seen_pairs.append((p1,p2))
    c1, c2 = one_p_over(p1, p2, True)
    c1 = mutation(c1, True)
    c2 = mutation(c2, True)

    next_gen.append(c1)
    next_gen.append(c2)

  next_gen.append(sorted_population[0][0])
  next_gen.append(sorted_population[1][0])
  return next_gen

def roulette_selection(totals):
  random_weight = random.random() * totals[-1]

  for index, weight in enumerate(totals):
    if random_weight < weight:
      return index

def mating_pool(sorted_pop, mating_pool_size):
  total_fitness = sum(x[1] for x in sorted_pop)
  weights = list(map(lambda x: x[1]/total_fitness, sorted_pop))
  totals = []
  for i in range(len(weights)):
    if i == 0:
      totals.append(weights[0])
    else:
      totals.append(weights[i] + totals[i-1])
  
  mating_pool = []
  while len(mating_pool) < mating_pool_size:
    random_chrom = roulette_selection(totals)
    if sorted_pop[random_chrom] in mating_pool:
      continue
    mating_pool.append(sorted_pop[random_chrom])
  return mating_pool

def sorted_pop(population):
  temp = []
  for i in range(len(population)):
    temp.append([population[i], fitness(population[i])])
  
  ret = sorted(temp, key=lambda x: x[1], reverse=True)
  return ret

def one_p_over(parent1, parent2, should_check_prob):
  if random.random() > p_c and should_check_prob:
    return parent1[0], parent2[0]
  
  index = random.randint(0, len(parent1[0]) -1)
  chrom = parent1[0][:index] + parent2[0][index:]
  chrom2 = parent2[0][:index] + parent1[0][index:]

  if is_valid(chrom) and is_valid(chrom2):
    return chrom, chrom2
  else:
    return one_p_over(parent1, parent2, False)

def mutation(chrom, should_check_prob):
  if random.random() > p_m and should_check_prob:
    return chrom

  index = random.randint(0, len(chrom) -1)
  flip = "1" if chrom[index] == "0" else "0"
  chrom2 = chrom[:index] + flip + chrom[index+1:]

  if is_valid(chrom2):
    return chrom2
  else:
    return mutation(chrom, False)

def get_params(chrom):
  pos1 = num_bits[0]
  pos2 = num_bits[0] + num_bits[1]

  args = [
    round(int(chrom[0:pos1], 2) / 100 + parameter_interval[0][0], 2),
    round(int(chrom[pos1: pos2], 2) / 100 + parameter_interval[1][0], 2),
    round(int(chrom[pos2:], 2) / 100 + parameter_interval[2][0], 2)
  ]

  return args

def is_valid(chrom):
  pos1 = num_bits[0]
  pos2 = num_bits[0] + num_bits[1]

  args = [
    round(int(chrom[0:pos1], 2) / 100 + parameter_interval[0][0], 2),
    round(int(chrom[pos1: pos2], 2) / 100 + parameter_interval[1][0], 2),
    round(int(chrom[pos2:], 2) / 100 + parameter_interval[2][0], 2)
  ]

  return (
    args[0] > parameter_interval[0][0] and args[0] < parameter_interval[0][1] and
    args[1] > parameter_interval[1][0] and args[1] < parameter_interval[1][1] and
    args[2] > parameter_interval[2][0] and args[2] < parameter_interval[2][1]
  )

def GA(num_iterations, initial_population=50):
  print("Starting GA")
  gen = initialize_population(initial_population)
  initial_pop_sorted = sorted_pop(gen)
  cur_best_chrom, cur_best_ise = initial_pop_sorted[0][0], initial_pop_sorted[0][1]
  print(cur_best_ise, cur_best_chrom, get_params(cur_best_chrom))
  best = [cur_best_ise]

  for i in range(num_iterations):
    gen = next_gen(gen, initial_population=initial_population)
    sorted_population = sorted_pop(gen)
    
    if sorted_population[0][1] > cur_best_ise:
      cur_best_chrom = sorted_population[0][0]
      cur_best_ise = sorted_population[0][1]

    best.append([cur_best_ise, cur_best_chrom])
    print(cur_best_ise, cur_best_chrom, get_params(cur_best_chrom))
  
  y = [best[0]]
  for i in range(1, len(best)):
    y.append(best[i][0])
  t = [i for i in range(num_iterations + 1)]

  return t,y

def part_c():
  t150, y150 = GA(150)

  plt.plot(t150, y150, label="150 iterations")
  plt.xlabel('Generation')
  plt.ylabel('Fitness Value')
  plt.title('Fitness vs Generation')
  plt.legend()
  plt.show()

def part_di():
  t25, y25 = GA(25)
  t50, y50 = GA(75)
  t175, y175 = GA(175)

  plt.plot(t25, y25, label="50 Generations")
  plt.plot(t50, y50, label="75 Generations")
  plt.plot(t175, y175, label="175 Generations")
  plt.xlabel('Generation')
  plt.ylabel('Fitness Value')
  plt.title('Fitness vs Generation')
  plt.legend()
  plt.show()

def part_dii():
  t10, y10 = GA(150, initial_population=10)
  t25, y25 = GA(150, initial_population=25)
  t60, y60 = GA(150, initial_population=60)

  plt.plot(t10, y10, label="Population Size 10")
  plt.plot(t25, y25, label="Population Size 25")
  plt.plot(t60, y60, label="Population Size 60")
  plt.xlabel('Generation')
  plt.ylabel('Fitness Value')
  plt.title('Fitness vs Generation')
  plt.legend()
  plt.show()

def part_diii():
  global p_m

  p_m = 0.1
  t, m_0_1 = GA(150)
  p_m = 0.4
  t, m_0_4 = GA(150)
  p_m = 0.7
  t, m_0_7 = GA(150)

  plt.plot(t, m_0_1, label="Mutation Probability 10%")
  plt.plot(t, m_0_4, label="Mutation Probability 40%")
  plt.plot(t, m_0_7, label="Mutation Probability 70%")
  plt.xlabel('Generation')
  plt.ylabel('Fitness Value')
  plt.title('Fitness vs Generation')
  plt.legend()
  plt.show()

def part_div():
  global p_c

  p_c = 0.1
  t, c_0_1 = GA(150)
  p_c = 0.4
  t, c_0_4 = GA(150)
  p_c = 0.7
  t, c_0_7 = GA(150)

  plt.plot(t, c_0_1, label="Crossover Probability 10%")
  plt.plot(t, c_0_4, label="Crossover Probability 40%")
  plt.plot(t, c_0_7, label="Crossover Probability 70%")
  plt.xlabel('Generation')
  plt.ylabel('Fitness Value')
  plt.title('Fitness vs Generation')
  plt.legend()
  plt.show()

def main():
  part_diii()
    
if __name__ == '__main__':
    main()