import numpy as np


def cost(ctx, X, R, S, Ps, Rt, gammat, **kwargs):
  M = X - R - S
  Et = np.sum(M * (ctx.P * ctx.dt)[None, :, None], axis=1)
  Pleft = ctx.Ptres - Rt - (1 - gammat) * Ps
  return np.sum((ctx.tau * Et) - (ctx.tau0 * Pleft), axis=-1)


def discomfort(ctx, X, R, S, **kwargs):
  Diff = (X - ctx.baseline)**2
  Dnrg = Diff * (ctx.P * ctx.dt)[None, :, None]
  Dwt = Dnrg[:, ctx.indxWperA] * ctx.xi[None, :, None]
  Dt = np.sum(Dwt, where=ctx.maskTperW, axis=1)
  return np.sum(Dt * ctx.tau, axis=-1)


def res_unsold(ctx, X, R, S, Ps, Rt, gammat, **kwargs):
  Pleft = ctx.Ptres - Rt - (1 - gammat) * Ps
  return -np.sum(ctx.tau0 * Pleft**2, axis=-1)


def res_waste(ctx, X, R, S, Ps, Rt, gammat, **kwargs):
  Pleft = ctx.Ptres - Rt - (1 - gammat) * Ps
  return np.sum(Pleft**2 * ctx.tau * 0.4, axis=-1)
