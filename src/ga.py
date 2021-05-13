import sys, argparse

import numpy as np
import pandas as pd

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling, get_termination
from pymoo.optimize import minimize
from pymoo.model.problem import Problem

from global_problem import GlobalProblem
from nsga2_int import IntegerUniformMutation


def run(demands, pop_size, **kwargs):
  algorithm = get_algorithm(
      'ga',
      sampling=get_sampling('int_random'),
      # crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
      crossover=get_crossover('int_hux'),
      # mutation=get_mutation("int_pm", eta=1.0),
      mutation=IntegerUniformMutation(),
      eliminate_duplicates=True,
      pop_size=pop_size,
  )
  instance = GlobalProblem(demands)

  if 'termination' in kwargs and kwargs['termination'] is not None:
    kwargs['termination'] = get_termination(*kwargs['termination'])
  else:
    kwargs['termination'] = None

  res = minimize(
      instance,
      algorithm,
      **kwargs,
      save_history=True,
  )

  if res.opt is None:
    return res, None

  return res, instance.data_from_history(res)
