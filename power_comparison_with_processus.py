from dataclasses import make_dataclass, field, astuple
from functools import partial, reduce
from multiprocessing import Pool

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import boschloo_exact, fisher_exact, barnard_exact

FUNCTIONS = [boschloo_exact, fisher_exact, barnard_exact]


def get_binom_samples(n, p, size=1000):
    tables = stats.binom.rvs(n, p, size=size)
    return tables


ResPvalues = make_dataclass('ResPvalues',
                            [('boschloo_exact', float, field(default=0)),
                             ('barnard_exact', float, field(default=0)),
                             ('fisher_exact', float, field(default=0)), ], )


def get_pvalues(table, alt="two-sided"):
    res = ResPvalues()
    for fn in FUNCTIONS:
        if fn.__name__ == "fisher_exact":
            _, res.fisher_exact = fisher_exact(table, alternative=alt)

        else:
            setattr(res, fn.__name__, fn(table, alternative=alt).pvalue)
    return astuple(res)


def count_threshold(map_obj, alpha):
    def cond(res: ResPvalues):
        """
        Condition if we accept H_0 or not
        """
        boschloo = 1 if res.boschloo_exact > alpha else 0
        barnard = 1 if res.barnard_exact > alpha else 0
        fisher = 1 if res.fisher_exact > alpha else 0
        return np.array([boschloo, barnard, fisher])

    return list(
        reduce(lambda x, y: x + y, map(cond, map_obj), np.array([0, 0, 0]))
    )


def mapping_samples(n1, n2, sample_tuple):
    a, b = sample_tuple
    return get_pvalues([[a, b], [n1 - a, n2 - b]], alt="less")


def count_rejection_hypothesis_with_workers(alpha=.05, repeat=1):
    for _ in range(repeat):
        # p_1 = .2 < p_2 = .3. Therefore, H_0 is false
        n1, p1, n2, p2 = 33, .2, 62, .3
        size = 100
        left_side = get_binom_samples(n1, p1, size)
        right_side = get_binom_samples(n2, p2, size)
        mapping_fn = partial(mapping_samples, n1, n2)

        # Let's compute power = 1 - 𝛽. Since 𝛽 = P(type II error) = P_Ha(
        # Accept H_0)
        with Pool(processes=None) as p:
            print(list(tqdm(p.imap(mapping_fn, zip(left_side, right_side)),
                         total=size)))
        # map_obj = list(map(mapping_fn, zip(left_side, right_side)))
        # print(count_treshold(map_obj, alpha=alpha))
        return

