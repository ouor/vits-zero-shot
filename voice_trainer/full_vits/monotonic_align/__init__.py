import numpy as np
import torch

try:
  from .core import maximum_path_c
except ImportError:
  maximum_path_c = None


def _maximum_path_python(neg_cent, mask):
  device = neg_cent.device
  dtype = neg_cent.dtype
  values = neg_cent.detach().cpu().numpy().astype(np.float32)
  masks = mask.detach().cpu().numpy().astype(np.float32)
  paths = np.zeros_like(values, dtype=np.int32)

  batch_size = values.shape[0]
  for batch_index in range(batch_size):
    t_t = int(masks[batch_index].sum(axis=1)[:, 0].max())
    t_s = int(masks[batch_index].sum(axis=0)[0].max())
    if t_t <= 0 or t_s <= 0:
      continue

    dp = np.full((t_t, t_s), -1e9, dtype=np.float32)
    prev = np.zeros((t_t, t_s), dtype=np.int32)
    dp[0, 0] = values[batch_index, 0, 0]

    for y in range(1, t_t):
      dp[y, 0] = dp[y - 1, 0] + values[batch_index, y, 0]

    for y in range(1, t_t):
      x_start = max(1, t_s + y - t_t)
      x_end = min(t_s, y + 1)
      for x in range(x_start, x_end):
        stay = dp[y - 1, x]
        move = dp[y - 1, x - 1]
        if move >= stay:
          dp[y, x] = move + values[batch_index, y, x]
          prev[y, x] = 1
        else:
          dp[y, x] = stay + values[batch_index, y, x]

    x = t_s - 1
    for y in range(t_t - 1, -1, -1):
      paths[batch_index, y, x] = 1
      if y > 0 and prev[y, x] == 1:
        x -= 1

  return torch.from_numpy(paths).to(device=device, dtype=dtype)


def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  if maximum_path_c is None:
    return _maximum_path_python(neg_cent, mask)

  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)
