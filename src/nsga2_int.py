import sys
import traceback
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination, get_selection
from pymoo.optimize import minimize
from pymoo.model.callback import Callback
from pymoo.model.mutation import Mutation
from pymoo.operators.integer_from_float_operator import IntegerFromFloatMutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

import problem_int as problem


class CustomMaximumFunctionCallTermination(MaximumFunctionCallTermination):
  def __init__(self, n_max_evals, n_max_inf_tol):
    super().__init__(n_max_evals)

    self.n_max_inf_tol = n_max_inf_tol

    if self.n_max_inf_tol is None:
      self.n_max_inf_tol = float("inf")

  def _do_continue(self, algorithm):
    feasible = algorithm.pop.get('feasible').flatten()
    if np.all(feasible):
      return super()._do_continue(algorithm)

    return algorithm.evaluator.n_eval < self.n_max_inf_tol and super(
    )._do_continue(algorithm)


class IntegerUniformMutation(Mutation):
  def __init__(self, prob=None):
    super().__init__()

    if prob is not None:
      self.prob = float(prob)
    else:
      self.prob = None

  def _do(self, problem, X, **kwargs):
    X = X.astype(np.int)
    Y = np.full(X.shape, np.inf)

    if self.prob is None:
      self.prob = 1.0 / problem.n_var

    do_mutation = np.random.random(X.shape) < self.prob

    Y[:, :] = X

    xr = np.array(
        [
            np.arange(problem.xl[i], problem.xu[i] + 1)
            for i in range(problem.xl.shape[0])
        ],
        dtype=np.int
    )
    xv = np.repeat(xr[None, :], X.shape[0], axis=0)[do_mutation]
    xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
    xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

    X = X[do_mutation]

    _Y = np.array([np.random.choice(xv[i][:]) for i in range(X.shape[0])])

    # back in bounds if necessary (floating point issues)
    indL = _Y < xl
    _Y[indL] = xl[indL]
    indU = _Y > xu
    _Y[indU] = xu[indU]

    # set the values for output
    Y[do_mutation] = _Y

    # in case out of bounds repair (very unlikely)
    Y = set_to_bounds_if_outside_by_problem(problem, Y)

    return Y


class CustomCallback(Callback):
  def __init__(self):
    super().__init__()
    self.data['gbest'] = []

  def notify(self, algorithm):
    pop = algorithm.pop
    feasible = pop.get('feasible').flatten()

    if not np.any(feasible):
      return

    objectives = pop.get('F')
    idx = np.argwhere(feasible)
    itn = algorithm.evaluator.n_eval
    ideal = np.min(objectives[idx], axis=0)

    if len(self.data['gbest']) == 0:
      self.data['gbest'].append(np.insert(ideal, 0, itn))
      return

    gbest = self.data['gbest'][-1][1:]

    if np.any(ideal < gbest):
      nideal = np.minimum(ideal, gbest)
      self.data['gbest'].append(np.insert(nideal, 0, itn))


def run(
    appliances,
    preferences,
    storage,
    generation,
    tariffs,
    sale=None,
    objs=None,
    population=100,
    offsprings=None,
    termination=None,
    mutation=None,
    crossover=None,
    sampling=None,
    sampling_seed=None,
    **kwargs
):
  if sale is None:
    sale = 0.5 * tariffs
  elif isinstance(sale, (int, float)):
    sale = sale * tariffs

  instance = problem.Scheduling(
      appliances=appliances,
      preferences=preferences,
      storage=storage,
      generation=generation,
      tariffs=tariffs,
      sale=sale,
      objectives=objs
  )

  if type(sampling) is dict:
    sampling = get_sampling(**sampling)
  elif type(sampling) in (tuple, list):
    sampling = get_sampling(*sampling)
  else:
    if sampling_seed is None and 'seed' in kwargs:
      sampling_seed = kwargs['seed']

    from numpy.random import PCG64, SeedSequence

    if sampling_seed is not None:
      rng = np.random.default_rng(PCG64(SeedSequence(sampling_seed)))
    else:
      rng = np.random.default_rng()
      sampling_seed = rng.bit_generator._seed_seq.entropy

    # converting baseline from AxT to WxT to get the actual decision variables by mask
    baseline = instance.baseline[instance.indxWperA][instance.maskTperW]
    sampling = rng.choice(
        [1, 2, 3], size=(population, *baseline.shape)
    ) * baseline

  if type(mutation) is dict:
    mutation = get_mutation(**mutation)
  elif type(mutation) in (tuple, list):
    mutation = get_mutation(*mutation)
  else:
    mutation = IntegerUniformMutation()

  if type(crossover) is dict:
    crossover = get_crossover(**crossover)
  elif type(crossover) in (tuple, list):
    crossover = get_crossover(*crossover)
  else:
    crossover = get_crossover('int_ux')

  if type(termination) is dict:
    termination = get_termination(**termination)
  elif type(termination) in (tuple, list):
    termination = get_termination(*termination)
  else:
    termination = CustomMaximumFunctionCallTermination(100_000, 35_000)

  algorithm = NSGA2(
      pop_size=population,
      n_offsprings=offsprings,
      sampling=sampling,
      mutation=mutation,
      crossover=crossover,
      eliminate_duplicates=True
  )

  res = None
  try:
    res = minimize(
        instance,
        algorithm,
        termination=termination,
        callback=CustomCallback(),
        **kwargs
    )
  except:
    traceback.print_exc()
    sys.exit(1)

  if res.opt is None:
    return res, None

  pf = res.opt
  x = pf.get('X')
  F = pf.get('F')
  I = F.shape[0]
  obj_names = [of.__name__ for of in res.problem.objs]

  X, R, S = instance.get_matrices(x)
  Es, Ps, Rt, St, gt = instance.calculate_der(R, S)

  dt = instance.dt
  P = instance.P

  Xta = np.transpose(X, axes=(0, 2, 1))
  Rta = np.transpose(R, axes=(0, 2, 1))
  Sta = np.transpose(S, axes=(0, 2, 1))

  appl_names = instance.appliances['label'].to_numpy()
  appl_energy = P * dt
  res_gen = instance.Ptres

  end_time = datetime.datetime.fromtimestamp(res.end_time)
  start_time = datetime.datetime.fromtimestamp(res.start_time)

  exec_recs = {
      'pop_size': [res.algorithm.pop_size],
      'n_offsprings': [res.algorithm.n_offsprings],
      'n_gen': [res.algorithm.n_gen],
      'n_eval': [res.algorithm.evaluator.n_eval],
      'seed': [res.algorithm.seed],
      'start_time': [start_time],
      'end_time': [end_time],
      'duration': [end_time - start_time],
  }

  exec_recs['sampling'] = getattr(
      res.algorithm.initialization.sampling, 'sampling',
      res.algorithm.initialization.sampling
  )
  exec_recs['sampling'] = [type(exec_recs['sampling']).__name__]
  exec_recs['sampling_seed'] = [str(sampling_seed)]

  exec_recs['mutation'] = getattr(
      res.algorithm.mating.mutation, 'mutation', res.algorithm.mating.mutation
  )
  exec_recs['mutation_prob'] = [getattr(exec_recs['mutation'], 'prob', 'None')]
  exec_recs['mutation_eta'] = [getattr(exec_recs['mutation'], 'eta', 'None')]
  exec_recs['mutation'] = [type(exec_recs['mutation']).__name__]

  exec_recs['crossover'] = getattr(
      res.algorithm.mating.crossover, 'crossover',
      res.algorithm.mating.crossover
  )
  exec_recs['crossover_prob'] = [
      getattr(exec_recs['crossover'], 'prob', 'None')
  ]
  exec_recs['crossover_eta'] = [getattr(exec_recs['crossover'], 'eta', 'None')]
  exec_recs['crossover'] = [type(exec_recs['crossover']).__name__]

  exec_recs['termination'] = getattr(
      res.algorithm.termination, 'termination', res.algorithm.termination
  )
  exec_recs['termination'] = [type(exec_recs['termination']).__name__]

  summary = {}
  summary['execution'] = pd.DataFrame.from_dict(
      exec_recs, orient='index', columns=['value']
  )

  summary['solution'] = pd.DataFrame(F, columns=obj_names)

  if res.algorithm.callback.data['gbest']:
    gbest = np.array(res.algorithm.callback.data['gbest'])
    summary['ideal'] = pd.DataFrame(gbest, columns=['it', *obj_names]).astype(
        {
            'it': 'int32'
        }
    ).set_index('it')

  keys = ('x', 'r', 's', 'm')
  summary['schedule'] = {k: [] for k in keys}
  summary['demand'] = {k: [] for k in keys}
  summary['profile'] = {k: [] for k in keys}

  xsch, rsch, ssch, msch = (summary['schedule'][k] for k in keys)
  xdmd, rdmd, sdmd, mdmd = (summary['demand'][k] for k in keys)
  xprf, rprf, sprf, mprf = (summary['profile'][k] for k in keys)
  for i in range(I):
    xschdf = pd.DataFrame(Xta[i], columns=appl_names)
    xsch.append(xschdf)

    rschdf = pd.DataFrame(Rta[i], columns=appl_names)
    rsch.append(rschdf)

    sschdf = pd.DataFrame(Sta[i], columns=appl_names)
    ssch.append(sschdf)

    mschdf = xschdf - rschdf - sschdf
    msch.append(mschdf)

    xdmddf = xschdf * appl_energy
    xdmddf['Sum'] = xdmddf.sum(axis=1, numeric_only=True)
    xprf.append(xdmddf['Sum'].to_numpy())
    xdmddf.loc['Total'] = xdmddf.sum(axis=0, numeric_only=True)
    xdmd.append(xdmddf)

    sdmddf = sschdf * appl_energy
    sdmddf['Sum'] = sdmddf.sum(axis=1, numeric_only=True)
    sprf.append(sdmddf['Sum'].to_numpy())
    sdmddf['Output'] = gt[i] * Ps[i]
    sdmddf['Charge'] = (1 - gt[i]) * Ps[i]
    sdmddf['Capacity'] = Es[i]
    sdmddf.loc['Total'] = sdmddf.sum(axis=0, numeric_only=True)
    sdmd.append(sdmddf)

    rdmddf = rschdf * appl_energy
    rdmddf['Sum'] = rdmddf.sum(axis=1, numeric_only=True)
    rprf.append(rdmddf['Sum'].to_numpy())
    rdmddf['Output'] = res_gen
    rdmddf['Spare'] = (rdmddf['Output'] *
                       dt) - (rdmddf['Sum'] + sdmddf['Charge'])
    rdmddf.loc['Total'] = rdmddf.sum(axis=0, numeric_only=True)
    rdmd.append(rdmddf)

    mdmddf = mschdf * appl_energy
    mdmddf['Sum'] = mdmddf.sum(axis=1, numeric_only=True)
    mprf.append(mdmddf['Sum'].to_numpy())
    mdmddf.loc['Total'] = mdmddf.sum(axis=0, numeric_only=True)
    mdmd.append(mdmddf)

  for k in keys:
    summary['profile'][k] = np.array(summary['profile'][k])

  return res, summary
