import numpy as np
import pandas as pd

from pymoo.model.problem import Problem


class GlobalProblem(Problem):
  def __init__(self, demands):
    # list of demand profiles to combine
    self.demands = demands

    # number of variables or customers
    self.N = len(demands)
    # number of profiles per customer
    self.M = [profiles.shape[0] - 1 for profiles in demands]
    # number of time intervals of load profile
    self.T = demands[0].shape[1]

    super().__init__(
        n_var=self.N,
        n_obj=1,
        n_constr=0,
        xl=np.zeros(self.N, dtype=np.int64),
        xu=np.array(self.M, dtype=np.int64),
        type_var=np.int64
    )

  def _evaluate(self, x, out, *args, **kwargs):
    profiles = self.x_to_profile(x)

    peaks = profiles.max(axis=1)
    means = np.sum(profiles, axis=1) / self.T

    out['F'] = peaks / means

  def x_to_profile(self, x):
    if x.dtype.type is not np.int64:
      x = self.to_int(x)

    profiles = np.zeros((x.shape[0], self.T))
    for i in range(self.N):
      loads = self.demands[i].astype(float)
      profiles = profiles + loads[x[:, i]]

    return profiles

  def to_int(self, x):
    return np.rint(x).astype(np.int64)

  def data_from_history(self, res):
    selections = self.to_int(res.X[None, :])
    profiles = self.x_to_profile(res.X[None, :])

    gbests = []
    for e in res.history:
      n_eval = e.evaluator.n_eval
      n_gen = e.n_gen
      ibest = e.opt[0].F.min()

      if not gbests:
        gbests.append((n_gen, n_eval, ibest))
        continue
      else:
        gbest = gbests[-1][-1]

      if ibest < gbest:
        gbests.append((n_gen, n_eval, ibest))

    profile = pd.DataFrame(data=profiles.T, columns=['value'])
    gbest = pd.DataFrame(data=gbests, columns=['it', 'neval', 'gb'])
    solution = pd.DataFrame(data=selections.T, columns=['solution'])
    summary = gbest.agg(['min', 'max', 'mean', 'std'])
    return {
        'profile': profile,
        'gbest': gbest,
        'solution': solution,
        'summary': summary
    }


class GlobalProblemNorm(Problem):
  def __init__(self, demands):
    # list of demand profiles to combine
    self.demands = demands

    # number of variables or customers
    self.N = len(demands)
    # number of profiles per customer
    self.M = np.array(
        [profiles.shape[0] - 1 for profiles in demands], dtype=np.int64
    )
    # number of time intervals of load profile
    self.T = demands[0].shape[1]

    super().__init__(
        n_var=self.N,
        n_obj=1,
        n_constr=0,
        xl=np.zeros(self.N),
        xu=np.ones(self.N),
        type_var=float
    )

  def _evaluate(self, x, out, *args, **kwargs):
    profiles = self.x_to_profile(x)

    peaks = profiles.max(axis=1)
    means = np.sum(profiles, axis=1) / self.T

    out['F'] = peaks / means

  def x_to_profile(self, x):
    x = self.to_int(x)

    profiles = np.zeros((x.shape[0], self.T))
    for i in range(self.N):
      loads = self.demands[i].astype(float)
      profiles = profiles + loads[x[:, i]]

    return profiles

  def to_int(self, x):
    return np.rint(x * self.M).astype(np.int64)

  def data_from_history(self, res):
    selections = self.to_int(res.X[None, :])
    profiles = self.x_to_profile(res.X[None, :])

    gbests = []
    for e in res.history:
      n_eval = e.evaluator.n_eval
      n_gen = e.n_gen
      ibest = e.opt[0].F.min()

      if not gbests:
        gbests.append((n_gen, n_eval, ibest))
        continue
      else:
        gbest = gbests[-1][-1]

      if ibest < gbest:
        gbests.append((n_gen, n_eval, ibest))

    profile = pd.DataFrame(data=profiles.T, columns=['value'])
    gbest = pd.DataFrame(data=gbests, columns=['it', 'neval', 'gb'])
    solution = pd.DataFrame(data=selections.T, columns=['solution'])
    summary = gbest.agg(['min', 'max', 'mean', 'std'])
    return {
        'profile': profile,
        'gbest': gbest,
        'solution': solution,
        'summary': summary
    }
