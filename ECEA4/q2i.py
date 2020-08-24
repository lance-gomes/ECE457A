import copy
import random
import matplotlib.pyplot as plt
from itertools import product 
from math import *
from enum import Enum

MAX_DEPTH = 6

class NT(Enum):
  a0 = 0
  a1 = 1
  a2 = 2
  d0 = 3
  d1 = 4
  d2 = 5
  d3 = 6
  d4 = 7
  d5 = 8
  d6 = 9
  d7 = 10

class NF(Enum):
  AND = 0
  OR = 1
  NOT = 2
  IF = 3

class Terminal:
  def __init__(self, node_type):
    self.node_type = node_type
  
  def depth(self):
    return 1

class Func:
  def __init__(self, node_type):
    self.node_type = node_type
    self.children = []
  
  def do_func(self, children_val):

    if self.node_type == NF.AND:
      return AND(children_val[0], children_val[1])
    elif self.node_type == NF.OR:
      return OR(children_val[0], children_val[1])
    elif self.node_type == NF.NOT:
      return NOT(children_val[0])
    else:
      return IF(children_val[0], children_val[1], children_val[2])

  def depth(self):
    max_depth = 1
    stack = []
    for i in self.children:
      stack.append([i, max_depth + 1])
    
    while len(stack) > 0:
      item = stack.pop(0)
      max_depth = max(max_depth, item[1])

      if is_func(item[0]):
        for j in item[0].children:
          stack.append([j, max_depth + 1])
    
    return max_depth

def AND(arg1, arg2):
  return arg1 and arg2

def OR(arg1, arg2):
  return arg1 or arg2

def NOT(arg1):
  if arg1 == 0:
    return 1
  else:
    return 0

def IF(arg1, arg2, arg3):
  if arg1:
    return arg2
  else:
    return arg3

def is_func(node):
  return isinstance(node, Func)

def eval_tree(node, inputs):
  # inputs = [ 0, 0, 1, 0, 0, 0 ]
  def helper(node, inputs):
    if is_func(node):
      vals = []
      for child in node.children:
        val = helper(child, inputs)
        vals.append(val)
      return node.do_func(vals)
    else:
      # return the input value associated with the node
      return inputs[node.node_type.value]
  return helper(node, inputs)

def rand_func_type():
  rnd = random.randint(0,3)

  if rnd == 0:
    return NF.AND
  elif rnd == 1:
    return NF.OR
  elif rnd == 2:
    return NF.NOT
  else:
    return NF.IF

def rand_terminal_type():
  rnd = random.randint(0,10)

  if rnd == 0:
    return NT.a0
  elif rnd == 1:
    return NT.a1
  elif rnd == 2:
    return NT.a2
  elif rnd == 3:
    return NT.d0
  elif rnd == 4:
    return NT.d1
  elif rnd == 5:
    return NT.d2
  elif rnd == 6:
    return NT.d3
  elif rnd == 7:
    return NT.d4
  elif rnd == 8:
    return NT.d5
  elif rnd == 9:
    return NT.d6
  elif rnd == 10:
    return NT.d7
    

def rand_func():
  node_type = rand_func_type()
  node = Func(node_type)

  children = []
  if node_type == NF.NOT:
    children.append(Terminal(rand_terminal_type()))
  elif node_type == NF.IF:
    children.append(Terminal(rand_terminal_type()))
    children.append(Terminal(rand_terminal_type()))
    children.append(Terminal(rand_terminal_type()))
  else:
    children.append(Terminal(rand_terminal_type()))
    children.append(Terminal(rand_terminal_type()))
  
  node.children = children
  return node

def rand_term():
  node_type = rand_terminal_type()
  node = Terminal(node_type)
  return node

def rand_tree_full(pop_size):
  ret = []
  for i in range(pop_size):
    node = rand_func() # step 1
    # too lazy to recursive, just gonna hardcode it
    replace_children(node) # step 2

    for i in node.children: # step 3
      replace_children(i)
    
    ret.append(node)
  return ret

def rand_node_in_tree(node):
  stack = []
  ret = []
  stack.append(node)

  while len(stack) > 0:
    n = stack.pop(0)
    if is_func(n):
      stack.extend(n.children)
    ret.append(n)

  return random.choice(ret)

def replace_node_in_tree(parent, node1, node2):
  stack = []
  stack.append(parent)

  while len(stack) > 0:
    n = stack.pop(0)
    if is_func(n):
      stack.extend(n.children)
      for i in range(len(n.children)):
        if n.children[i] is node1:
          n.children[i] = node2
          return parent
    
    if n is node1:
      n = node2
      return parent

def rand_tree_grow(pop_size):
  ret = []
  for i in range(pop_size):
    node = rand_term()
    ret.append(node)
  return ret

def replace_children(node):
  for i in range(len(node.children)):
    node.children[i] = rand_func()

# basically just get all combinations of 0,1 in array of size 11
def tests():
  tests = product('01', repeat=11)
  tests = (list(tests))
  ret = []
  for i in tests:
    l = []
    for j in i:
      l.append(int(j))
    ret.append(l)
  return ret

def run_tests(node, t):
  fitness = 0
  for test in t:
    result = eval_tree(node, test)
    if test[0] == 0 and test[1] == 0 and test[2] == 0 and ((result == 1 and test[3] == 1) or (result == 0 and test[3] == 0)):
      fitness +=1
    elif test[0] == 0 and test[1] == 0 and test[2] == 1 and ((result == 1 and test[4] == 1) or (result == 0 and test[4] == 0)):
      fitness +=1
    elif test[0] == 0 and test[1] == 1 and test[2] == 0 and ((result == 1 and test[5] == 1) or (result == 0 and test[5] == 0)):
      fitness +=1
    elif test[0] == 0 and test[1] == 1 and test[2] == 1 and ((result == 1 and test[6] == 1) or (result == 0 and test[6] == 0)):
      fitness +=1
    elif test[0] == 1 and test[1] == 0 and test[2] == 0 and ((result == 1 and test[7] == 1) or (result == 0 and test[7] == 0)):
      fitness +=1
    elif test[0] == 1 and test[1] == 0 and test[2] == 1 and ((result == 1 and test[8] == 1) or (result == 0 and test[8] == 0)):
      fitness +=1
    elif test[0] == 1 and test[1] == 1 and test[2] == 0 and ((result == 1 and test[9] == 1) or (result == 0 and test[9] == 0)):
      fitness +=1
    elif test[0] == 1 and test[1] == 1 and test[2] == 1 and ((result == 1 and test[10] == 1) or (result == 0 and test[10] == 0)):
      fitness +=1
  return fitness

def population(pop_size, t):
  pops = []
  ret = []

  p1 = rand_tree_full(pop_size // 2)
  p2 = rand_tree_grow(pop_size // 2)

  pops.extend(p1)
  pops.extend(p2)

  for i in pops:
    fitness = run_tests(i, t)
    ret.append([i, fitness])

  return ret

def run_fitness(po, t):
  for p in po:
    fitness = run_tests(p[0], t)
    if len(p) == 2:
      p[1] = fitness
    elif len(p) == 1:
      p.append(fitness)
  return po

def tournament(pop):
  ret = []

  while len(ret) < 25:
    candidates = []

    while len(candidates) < 5:
      index = random.randint(0, len(pop) -1 )
      candidates.append([pop[index], index])

    sorted_candidates = sorted(candidates, key=lambda x: x[0][1], reverse=True)
    pop.pop(sorted_candidates[0][1])
    ret.append(sorted_candidates[0][0][0])
  
  return ret

def mating_pool(parents):
  ret = []

  while len(ret) < 100:
    p1 = copy.deepcopy(parents[random.randint(0, len(parents) - 1)])
    p2 = copy.deepcopy(parents[random.randint(0, len(parents) - 1)])

    if p1 == p2:
      continue

    if random.uniform(0,1) > 0.01:
      c1, c2 = crossover(p1, p2)
      if c1.depth() <= MAX_DEPTH:
        ret.append([c1])

      if c2.depth() <= MAX_DEPTH:
        ret.append([c2])
    else:
      ret.append([p1])
      ret.append([p2])
  
  return ret

def crossover(p1, p2):
  rand1 = rand_node_in_tree(p1)
  rand2 = rand_node_in_tree(p2)

  replacement_rand1 = copy.deepcopy(rand1)
  replacement_rand2 = copy.deepcopy(rand2)

  replace_node_in_tree(p1, rand1, replacement_rand2)
  replace_node_in_tree(p2, rand2, replacement_rand1)

  return p1, p2

def GP(num_iterations=100, size_pop=50):
  t = tests()
  p = population(size_pop, t)
  p = run_fitness(p, t)
  y = [sorted(p, key=lambda x: x[1], reverse=True)[0][1] / 2048]
  best = y[0]
  best_node = sorted(p, key=lambda x: x[1], reverse=True)[0][0]

  for i in range(num_iterations):
    tourn = tournament(p)
    p = mating_pool(tourn)
    p = run_fitness(p,t)

    best_pop = sorted(p, key=lambda x: x[1], reverse=True)
    best_fitness = best_pop[0][1] / 2048

    if best_fitness >= best:
      best = best_fitness
      best_node = best_pop[0][0]
    
    y.append(best)
    print(best)

  t = [i for i in range(len(y))]

  return t, y, best_node

def print_tree(node):

  cur_depth = 0
  stack = []
  line = ""
  print("Best Node Tree")
  print(str(node.node_type))

  for i in node.children:
    stack.append([i, cur_depth])

  while len(stack) > 0:
    n = stack.pop(0)
    if n[1] > cur_depth:
      print(line)
      line = str(n[0].node_type)
      cur_depth = n[1]
    else:
      line = line + "  " + str(n[0].node_type)
    
    if is_func(n[0]):
      for i in n[0].children:
        stack.append([i, cur_depth + 1])

  print(line)


def part1():
  t,y, best_node = GP()
  print_tree(best_node)

  plt.figure(1)
  plt.plot(t, y)
  plt.xlabel('Generation')
  plt.ylabel('Global Best Fitness')
  plt.title('Global Best Fitness Per Generation')
  plt.show()

  import pdb; pdb.set_trace()

def main():
  part1()

if __name__ == "__main__":
  main()





