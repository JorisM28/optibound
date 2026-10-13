import torch
import numpy as np


from functions import EllipticBowl
from estimator import MonteCarloLipschitz


cuv = EllipticBowl()

est = MonteCarloLipschitz(cuv, (-10, 10), 10000)

res = est.estimate(batch_size=2000, epsilon=1e-7)

print(res)

