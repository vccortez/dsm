import numpy as np
import pandas as pd

from pymoo.model.problem import Problem

import objectives as of


class Scheduling(Problem):
  """Household loads scheduling problem, version 3."""
  def __init__(
      self,
      appliances,
      preferences,
      storage,
      generation,
      tariffs,
      sale,
      objectives=None
  ):
    """Build a scheduling problem instance.

    Parameters:
    - appliances (pandas.DataFrame): `A×('id','power','label','description')`
    - preferences (pandas.DataFrame): `W×('appliance','alpha','omega','dmin','dcur','def','int','xi')`
    - storage (pandas.DataFrame): `('capacitymin','capacitymax','capacity0','powermax','efficiencyc','efficiencyd')`
    - generation (pandas.DataFrame): `T×(val)` generation outputs in W at each time slot
    - tariffs (pandas.DataFrame): `T×(val)` monetary costs per kW at each time slot
    - sale (pandas.DataFrame): `T×(val)` energy sale costs per kW at each time slot
    """
    self.appliances = appliances
    self.preferences = preferences
    self.storage = storage
    self.generation = generation
    self.tariffs = tariffs
    self.sale = sale

    # parameter extraction
    if objectives is None:
      self.objs = [of.cost, of.discomfort]
    else:
      self.objs = [f for f in objectives if callable(f)]

    # number of appliances
    self.sizeA = appliances.shape[0]
    # number of time slots
    self.sizeT = tariffs.shape[0]
    # time step size (hour)
    self.dt = 24 / self.sizeT
    # electricity price per time
    self.tau = tariffs['val'].to_numpy()
    # electricity sale price per time
    self.tau0 = sale['val'].to_numpy()
    # power consumption per appliance
    self.P = appliances['power'].to_numpy()

    # power generated per time step
    self.Ptres = generation['val'].to_numpy()
    # maximum revenue for all RES power
    self.RESmaxrev = np.sum(self.Ptres**2 * 0.4 * self.tau, axis=0)
    # ess parameters
    self.Esmin = storage.at[0, 'capacitymin']
    self.Esmax = storage.at[0, 'capacitymax']
    self.Es0 = storage.at[0, 'capacity0']
    self.Psmax = storage.at[0, 'powermax']
    self.efc = storage.at[0, 'efficiencyc']
    self.efd = storage.at[0, 'efficiencyd']

    appliances = appliances.reset_index().set_index('id')
    grouped_preferences = preferences.groupby('appliance')
    preference_index = preferences['appliance'].to_numpy()

    # total number of operation windows
    self.sizeW = np.sum(grouped_preferences.size().to_numpy())

    # reverse appliance index per window
    self.indxWperA = appliances.loc[preference_index, 'index'].to_numpy()
    # alpha values per window
    self.alpha = preferences['alpha'].to_numpy()
    # omega values per window
    self.omega = preferences['omega'].to_numpy()
    # mininum demand per window
    self.dmin = preferences['dmin'].to_numpy()
    # curtailable demand per window
    self.dcur = preferences['dcur'].to_numpy()
    # deferrable load per window
    self.isDef = preferences['def'].to_numpy()
    # interruptible load per window
    self.isInt = preferences['int'].to_numpy()
    # elasticity factor per window
    self.xi = preferences['xi'].to_numpy()
    # baseline matrix
    self.baseline = np.zeros((self.sizeA, self.sizeT), dtype=int)

    # number of time slots required for each window's time region
    slots = self.omega + 1 - self.alpha
    # total number of integer decision variables
    self.sizeN = np.sum(slots)

    # mask for time window region per appliance
    self.maskTperA = np.zeros((self.sizeA, self.sizeT), dtype=bool)
    # mask for time window region per window
    self.maskTperW = np.zeros((self.sizeW, self.sizeT), dtype=bool)
    # mask for decision variable region per window
    self.maskNperW = np.zeros((self.sizeW, self.sizeN), dtype=bool)

    # computing masks
    slotsum = np.cumsum(slots)
    for w in range(self.sizeW):
      beginT = 0 if w == 0 else slotsum[w - 1]
      endT = slotsum[w]
      self.maskNperW[w, beginT:endT] = True

      appl = self.indxWperA[w]
      dmin = self.dmin[w]
      beginT = self.alpha[w]
      endT = self.omega[w] + 1
      self.maskTperW[w, beginT:endT] = True
      self.maskTperA[appl, beginT:endT] = True
      self.baseline[appl, beginT:beginT + dmin] = 1

    # constraints counting
    constr = []
    # constraint 1: no operation outside appliances windows
    constr.append(self.sizeA)
    # constraint 2: no deferring operation
    constr.append(self.sizeW)
    # constraint 3: no interrupting operation
    constr.append(self.sizeW)
    # constraint 4: respect minimum demand
    constr.append(self.sizeW)
    # constraint 5a: auxiliary matrices upper bound
    constr.append(self.sizeA * self.sizeT)
    # constraint 5a: auxiliary matrices lower bound
    constr.append(self.sizeA * self.sizeT)
    # constraint 6: respect renewable power supply
    constr.append(self.sizeT)
    # constraint 7a: stored capacity lower bound
    constr.append(self.sizeT)
    # constraint 7b: stored capacity upper bound
    constr.append(self.sizeT)
    # constraint 8a: storage output lower bound
    constr.append(self.sizeT)
    # constraint 8b: storage output upper bound
    constr.append(self.sizeT)
    # constraint 9: respect stored power supply
    constr.append(self.sizeT)
    # constraint 10: respect renewable power supply when charging
    constr.append(self.sizeT)
    # constraint 11: initial storage equals final
    constr.append(1)

    self.gn = np.array(constr)

    super().__init__(
        n_var=self.sizeN,
        n_obj=len(self.objs),
        n_constr=np.sum(self.gn),
        xl=np.zeros(self.sizeN, dtype=int),
        xu=np.full(self.sizeN, 3, dtype=int),
        type_var=np.int,
        elementwise_evaluation=False
    )

  def _evaluate(self, x, out, *args, **kwargs):
    """Evaluate population of solutions.

    Parameters:
    - x (numpy.ndarray): population sized solutions with
      `n_var` decision variables
    - out (numpy.ndarray): objectives and constraints outputs
    """
    ipt = {}
    ipt['X'], ipt['R'], ipt['S'] = self.get_matrices(x)
    ipt['Es'], ipt['Ps'], ipt['Rt'], ipt['St'], ipt[
        'gammat'] = self.calculate_der(**ipt)

    fs = [fi(self, **ipt) for fi in self.objs]

    gs = self.calculate_constraints(**ipt)

    out['F'] = np.column_stack(fs)
    out['G'] = np.concatenate(gs, axis=1)

  def get_matrices(self, x):
    # number of solutions
    I = x.shape[0]

    # decision matrices
    X = np.zeros((I, self.sizeA, self.sizeT), dtype=int)
    R = np.zeros((I, self.sizeA, self.sizeT), dtype=int)
    S = np.zeros((I, self.sizeA, self.sizeT), dtype=int)

    # grabbing decision variables and indexing for each w
    vx = (x > 0).astype(int)
    vr = (x == 2).astype(int)
    vs = (x == 3).astype(int)
    for w in range(self.sizeW):
      mask = self.maskNperW[w]
      appl = self.indxWperA[w]
      beginT = self.alpha[w]
      endT = self.omega[w] + 1
      X[:, appl, beginT:endT] = vx[:, mask]
      R[:, appl, beginT:endT] = vr[:, mask]
      S[:, appl, beginT:endT] = vs[:, mask]

    return X, R, S

  def calculate_constraints(self, X, R, S, Es, Ps, Rt, St, gammat, **kwargs):
    I = X.shape[0]
    A = self.sizeA
    W = self.sizeW
    T = self.sizeT
    dt = self.dt

    dmin = self.dmin
    dcur = self.dcur

    E = self.P * dt
    Ta = self.maskTperA
    alpha = self.alpha

    Tw = self.maskTperW
    ldef = self.isDef
    lint = self.isInt

    Xwt = X[:, self.indxWperA].copy()
    for w in range(W):
      mask = Tw[w]
      data = Xwt[:, w]
      Xwt[:, w] = data * mask

    Xwt_ndef = Xwt.copy()
    Xwt_ndef[:, ldef] = 0

    Xwt_nint = Xwt.copy()
    Xwt_nint[:, lint] = 0

    def first_nonzero(arr, axis, invalid_val=-1):
      mask = arr != 0
      return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def last_nonzero(arr, axis, invalid_val=-1):
      mask = arr != 0
      val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
      return np.where(mask.any(axis=axis), val, invalid_val)

    gi = [None for _ in self.gn]
    # constraint 1: no operation outside appliance windows
    gi[0] = np.sum(X, axis=2, where=np.invert(Ta))
    assert gi[0].shape[1] == self.gn[0]

    alpha_plus = first_nonzero(Xwt_ndef, axis=2)
    # constraint 2: no deferring operation
    gi[1] = alpha_plus - alpha
    assert gi[1].shape[1] == self.gn[1]

    # constraint 3: no interrupting operation
    gi[2] = np.zeros(gi[1].shape)
    assert gi[2].shape[1] == self.gn[2]

    alpha_plus = first_nonzero(Xwt_nint, axis=2)
    omega_plus = last_nonzero(Xwt_nint, axis=2)

    for w in range(W):
      alphas = alpha_plus[:, w]
      omegas = omega_plus[:, w]

      for i in range(I):
        a = alphas[i]

        if a == -1:
          gi[2][i, w] = 0
        else:
          o = omegas[i] + 1
          prod = np.prod(Xwt_nint[i, w, a:o])
          gi[2][i, w] = -prod + 1

    daw = np.sum(Xwt, axis=2)
    # constraint 4: respect minimum demand
    gi[3] = -(daw - dmin + dcur)
    assert gi[3].shape[1] == self.gn[3]

    M = (X - R) - S
    m = np.reshape(M, (-1, A * T))
    # constraint 5a: auxiliary matrices upper bound
    gi[4] = m - 1
    assert gi[4].shape[1] == self.gn[4]

    # constraint 5b: auxiliary matrices lower bound
    gi[5] = -m
    assert gi[5].shape[1] == self.gn[5]

    Ptres = self.Ptres
    # constraint 6: respect renewable power supply
    gi[6] = Rt - (dt * Ptres)
    assert gi[6].shape[1] == self.gn[6]

    Esmin = self.Esmin
    # constraint 7a: stored capacity lower bound
    gi[7] = Esmin - Es
    assert gi[7].shape[1] == self.gn[7]

    Esmax = self.Esmax
    # constraint 7b: stored capacity upper bound
    gi[8] = Es - Esmax
    assert gi[8].shape[1] == self.gn[8]

    # constraint 8a: storage output lower bound
    gi[9] = -Ps
    assert gi[9].shape[1] == self.gn[9]

    Psmax = self.Psmax
    # constraint 8b: storage output upper bound
    gi[10] = Ps - Psmax
    assert gi[10].shape[1] == self.gn[10]

    # constraint 9: respect stored power supply
    gi[11] = St - (gammat * dt * Ps)
    assert gi[11].shape[1] == self.gn[11]

    # constraint 10: respect renewable power supply when charging
    gi[12] = (1 - gammat) * dt * Ps - (dt * Ptres - Rt)
    assert gi[12].shape[1] == self.gn[12]

    # constraint 11: initial storage equals final
    gi[13] = np.zeros((gi[0].shape[0], 1))
    gi[13][:, 0] = self.Es0 - Es[:, T - 1]
    assert gi[13].shape[1] == self.gn[13]

    return tuple(gi)

  def calculate_der(self, R, S, **kwargs):
    dt = self.dt
    Ptres = self.Ptres
    ec = self.efc
    ed = self.efd
    Es0 = self.Es0
    Esmax = self.Esmax
    Psmax = self.Psmax
    E = (self.P * dt)[None, :, None]

    Rt = np.sum(E * R, axis=1)
    St = np.sum(E * S, axis=1)
    gt = (St > 0).astype(int)

    Ps = np.zeros(gt.shape)
    Es = np.zeros(gt.shape)

    Pdis = St[:, 0] / dt
    Pmaxess = Ptres[0] - Rt[:, 0] / dt
    Pmaxcap = (Esmax - Es0) / ((1 - gt[:, 0]) * ec - gt[:, 0] / ed)
    Pcha = np.clip(np.minimum(Pmaxcap, Pmaxess), None, Psmax)
    Ps[:, 0] = gt[:, 0] * Pdis + (1 - gt[:, 0]) * Pcha

    Ed = dt * Ps[:, 0] * ((1 - gt[:, 0]) * ec - gt[:, 0] / ed)
    Es[:, 0] = Es0 + Ed

    for t in range(1, self.sizeT):
      Ep = Es[:, t - 1]
      Pdis = St[:, t] / dt
      Pmaxess = Ptres[t] - Rt[:, t] / dt
      Pmaxcap = (Esmax - Ep) / ((1 - gt[:, t]) * ec - gt[:, t] / ed)
      Pcha = np.clip(np.minimum(Pmaxcap, Pmaxess), 0, Psmax)
      Ps[:, t] = gt[:, t] * Pdis + (1 - gt[:, t]) * Pcha

      Ed = dt * Ps[:, t] * ((1 - gt[:, t]) * ec - gt[:, t] / ed)
      Es[:, t] = Ep + Ed

    return Es, Ps, Rt, St, gt
