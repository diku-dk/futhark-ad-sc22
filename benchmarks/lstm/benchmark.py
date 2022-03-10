from abc import ABC, abstractmethod
import torch

class Benchmark(ABC):
  @abstractmethod
  def __init__(self, runs, *args):
    self.runs = runs

  @abstractmethod
  def prepare(self):
      ...

  @abstractmethod
  def calculate_objective(self):
      ...

  @abstractmethod
  def calculate_jacobian(self):
      ...

  def time_fun(self, f):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(self.runs):
      f()
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / self.runs

  def time_objective(self):
      self.objective_time = self.time_fun(self.calculate_objective)

  def time_jacobian(self):
      self.jacobian_time = self.time_fun(self.calculate_jacobian)

  def benchmark(self):
    self.prepare()
    self.time_objective()
    self.time_jacobian()

  def report(self):
    return { 'objective': self.objective_time*1000,
             'jacobian': self.jacobian_time*1000,
             'overhead': self.jacobian_time/self.objective_time
           }

def set_precision(prec):
  if (prec == "f32"):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_dtype(torch.float32)
  elif (prec == "f64"):
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.set_default_dtype(torch.float64)
  else:
    sys.exit("Error: invalid precision " + prec)
  


