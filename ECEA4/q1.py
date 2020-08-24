import copy
import random
from math import *
import matplotlib.pyplot as plt

e_s = 15
e_f = 5

class Particle:
  def __init__(self, x, v, p_best, c1, c2, w=1):
    self.x = x
    self.v = v
    self.p_best = p_best
    self.c1 = c1
    self.c2 = c2
    self.w = w

  def next_step_simple(self, g_best):
    self.v[0] = self.v[0] + self.c1 * random.random() * (self.p_best[0] - self.x[0]) + self.c2 * random.random() * (g_best[0] - self.x[0])
    self.v[1] = self.v[1] + self.c1 * random.random() * (self.p_best[1] - self.x[1]) + self.c2 * random.random() * (g_best[1] - self.x[1])

    self.x[0] = self.x[0] + self.v[0]
    self.x[1] = self.x[1] + self.v[1]

    self.set_max()
    return self.solution()
  
  def next_step_inertia(self, g_best):
    self.v[0] = self.w * self.v[0] + self.c1 * random.random() * (self.p_best[0] - self.x[0]) + self.c2 * random.random() * (g_best[0] - self.x[0])
    self.v[1] = self.w * self.v[1] + self.c1 * random.random() * (self.p_best[1] - self.x[1]) + self.c2 * random.random() * (g_best[1] - self.x[1])
    self.x[0] = self.x[0] + self.v[0]
    self.x[1] = self.x[1] + self.v[1]

    self.set_max()
    return self.solution()

  def next_step_constriction(self, g_best):
    phi = self.c1 + self.c2
    K = 2 / (abs(2 - phi - sqrt(phi*phi - 4*phi)))

    self.v[0] = K * (self.v[0] + self.c1 * random.random() * (self.p_best[0] - self.x[0]) + self.c2 * random.random() * (g_best[0] - self.x[0]))
    self.v[1] = K * (self.v[1] + self.c1 * random.random() * (self.p_best[1] - self.x[1]) + self.c2 * random.random() * (g_best[1] - self.x[1]))
    self.x[0] = self.x[0] + self.v[0]
    self.x[1] = self.x[1] + self.v[1]

    self.set_max()
    return self.solution()

  def next_step_guaranteed(self, g_best, p_c, g_best_reference):

    if g_best_reference is self:
      self.v[0] = -self.v[0] + g_best[0] + self.w * self.v[0] + p_c * (1 - 2 * random.random())
      self.v[1] = -self.v[1] + g_best[1] + self.w * self.v[1] + p_c * (1 - 2 * random.random()) 
      self.x[0] = g_best[0] + self.w * self.v[0] + p_c * (1 - 2 * random.random())
      self.x[1] = g_best[1] + self.w * self.v[1] + p_c * (1 - 2 * random.random())
    else:
      self.v[0] = self.w * self.v[0] + self.c1 * random.random() * (self.p_best[0] - self.x[0]) + self.c2 * random.random() * (g_best[0] - self.x[0])
      self.v[1] = self.w * self.v[1] + self.c1 * random.random() * (self.p_best[1] - self.x[1]) + self.c2 * random.random() * (g_best[1] - self.x[1])
      self.x[0] = self.x[0] + self.v[0]
      self.x[1] = self.x[1] + self.v[1]

    self.set_max()
    return self.solution()
  
  def set_max(self):
    if self.x[0] <= -5:
      self.x[0] = -5

    if self.x[0] >= 5:
      self.x[0] = 5
      
    if self.x[1] <= -5:
      self.x[1] = -5

    if self.x[1] >= 5:
      self.x[1] = 5
  
  def solution(self):
    x_cord = self.x[0]
    y_cord = self.x[1]

    z =  ( (4 - 2.1 * pow(x_cord, 2) + pow(x_cord,4) / 3) * pow(x_cord, 2) )  + x_cord*y_cord + ((-4 + 4* pow(y_cord, 2))* pow(y_cord,2))
    self.x[2] = z

    if self.p_best == None or z < self.p_best[2]:
      self.p_best = copy.copy(self.x)

    return self.x

class Swarm:
  def __init__(self, c1, c2, w=1, step_type=0):
    self.step_type = step_type
    self.c1 = c1
    self.c2 = c2
    self.w = w
    self.parameter_control = 1.0
    self.num_success = 0
    self.num_failure = 0
  
  def initialize_population(self):
    p = []
    z = []
    
    for i in range(10):
      x = random.randint(-4, 5)
      y = random.randint(-4, 5)
      particle = Particle([x, y, None], [0,0], None, self.c1, self.c2, w=self.w)
      particle.solution()
      z.append(particle.x)
      p.append(particle)

    self.g_best = min(z, key=lambda x: x[2])
    self.g_best_reference = min(p, key=lambda x: x.x[2])
    self.p = p

  def update(self):
    r = []
    avg = 0

    for p in self.p:
      if self.step_type == 0:
        x = p.next_step_simple(self.g_best)
      elif self.step_type == 1:
        x = p.next_step_inertia(self.g_best)
      elif self.step_type == 2:
        x = p.next_step_constriction(self.g_best)
      elif self.step_type == 3:
        x = p.next_step_guaranteed(self.g_best, self.parameter_control, self.g_best_reference)

      r.append(x)
      avg += x[2]

    avg = avg / len(self.p)
    s_best = min(r, key=lambda x: x[2])
    s_best_ref = min(self.p, key=lambda x: x.x[2])

    if s_best[2] < self.g_best[2]:
      self.g_best = copy.copy(s_best)

      if self.g_best_reference is s_best:
        self.num_failure = 0
        self.num_success += 1
      else: 
        self.g_best_reference = s_best_ref
        self.num_failure = 0
        self.num_success = 0
        self.parameter_control = 1.0

    elif s_best[2] == self.g_best[2]:
      self.num_success = 0
      self.num_failure += 1

    else:
      self.num_failure = 0
      self.num_success += 1
    
    if self.num_success > e_s:
      self.parameter_control = 2 * self.parameter_control
    elif self.num_failure > e_f:
      self.parameter_control = 0.5 * self.parameter_control

    return self.g_best, avg

def PSO(c1=0.25, c2=0.25, w=1, step_type=0, iters=50):
  s = Swarm(c1, c2, w, step_type=step_type)
  s.initialize_population()
  y = []
  y1 = []

  for i in range(iters):
    best, avg = s.update()
    y.append(best[2])
    y1.append(avg)
  print("Final Solution: ", s.g_best)
  t = [i for i in range(len(y))]
  return t, y, y1

def part1():
  t, y, y1 = PSO()

  plt.figure(1)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Simple PSO Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(2)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Simple PSO Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(3)
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Simple PSO Hump Camelback Value vs Iteration')
  plt.legend()

  plt.show()

def part2():
  t, y, y1 = PSO(w=0.5, step_type=1)

  plt.figure(1)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Inertia Weight Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(2)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Inertia Weight Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(3)
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Inertia Weight Hump Camelback Value vs Iteration')
  plt.legend()

  plt.show()

def part3():
  t, y, y1 = PSO(step_type=2, c1=2.05, c2=2.05)

  plt.figure(1)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Constriction Factor Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(2)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Constriction Factor Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(3)
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Constriction Factor Hump Camelback Value vs Iteration')
  plt.legend()

  plt.show()

def part4():
  t, y, y1 = PSO(step_type=3, w=0.5)

  plt.figure(1)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Guaranteed Convergence Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(2)
  plt.plot(t, y, label="Best Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Guaranteed Convergence Hump Camelback Value vs Iteration')
  plt.legend()

  plt.figure(3)
  plt.plot(t, y1, label="Average Particle Fitness")
  plt.xlabel('Iteration')
  plt.ylabel('Z Value')
  plt.title('Guaranteed Convergence Hump Camelback Value vs Iteration')
  plt.legend()

  plt.show()

def main():
  part1()

if __name__ == "__main__":
  main()

