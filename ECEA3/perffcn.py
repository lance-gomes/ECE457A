import control

def tf(arg1, arg2):
  return control.TransferFunction(arg1, arg2)

def feedback(arg1, arg2):
  return control.feedback(arg1, arg2)

def step_info(arg1, arg2):
  return control.step_info(arg1, T=arg2)

def step_response(arg1, arg2):
  return control.step_response(arg1, T=arg2)

def series(arg1, arg2):
  return control.series(arg1, arg2)

def frange(start, stop, step):
  ret = []
  while start < stop:
    ret.append(start)
    start += step
  return ret

def perffcn(Kp, Ti, Td):
  G = Kp*tf([Ti*Td,Ti,1],[Ti,0])
  F = tf(1,[1,6,11,6,0])
  sys = feedback(series(G,F),1)
  t = frange(0, 100, 0.01)
  sysinf = step_info(sys, t)
  T,y = step_response(sys, t)
  return sum((y-1) **2), sysinf['RiseTime'], sysinf['SettlingTime'], sysinf['Overshoot']