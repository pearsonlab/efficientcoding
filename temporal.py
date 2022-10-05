import torch


def preset_temp(ntype):  # (8 mosaics) ver
    if ntype == 'slow':
        temp = torch.tensor(
            [1.35224712e-05, 4.31683979e-05, 1.22903181e-04, 3.00658601e-04,
             5.96876655e-04, 8.71045249e-04, 7.70543303e-04, 2.56589481e-04,
             1.53375523e-06, 0.00000000e+00])

    elif ntype == 'fastA':
        temp = torch.tensor(
            [-0.00018356, -0.00055406, -0.00144886, -0.00308577, -0.00471213,
             -0.00335751, 0.0033153, 0.0072599, 0.00143921, 0.])

    elif ntype == 'fastB':
        temp = -1 * torch.tensor(
            [-1.64128217e-06, -1.63703639e-05, -1.46979214e-04, -1.14184450e-03,
             -7.17923817e-03, -3.20750076e-02, -7.20159411e-02, 3.51153868e-02,
             1.31179707e-01, 0.00000000e+00])

    elif ntype == 'fastC':
        temp = torch.tensor(
            [-3.60435067e-15, -5.39885682e-13, -7.48634136e-11, -9.35443778e-09,
             -1.00630197e-06, -8.56234619e-05, -4.80940915e-03, -1.08762447e-01,
             1.05297446e-01, 0.00000000e+00])
    else:
        raise TypeError("ntype should be one of 'slow, fastA, fastB, or fastC")
    return temp / torch.norm(temp)
