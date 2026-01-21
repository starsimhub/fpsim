# fpsim/stockout.py

import numpy as np
from fpsim.interventions import Intervention
from fpsim.utils import bt


class StockoutIntervention(Intervention):
    """
    Discontinue method use if method-specific stockout occurs.
    Users specify year- and method-specific stockout probabilities.
    """

    def __init__(self, stockout_probs: dict[int, dict[int, float]], seed: int | None = None):
        super().__init__()
        self.stockout_probs = stockout_probs  # {year: {method_id: probability}}
        self.rng = np.random.default_rng(seed)

    def apply(self, sim):
        print(f"[Stockout] apply() called at year {sim.y:.2f}")

        year = int(sim.y)
        if year not in self.stockout_probs:
            return

        probs_for_year = self.stockout_probs[year]
        ppl = sim.people

        for i in range(len(ppl)):
            m = int(ppl.method[i])
            if m == 0:
                continue  # Not on a method

            p_stock = probs_for_year.get(m, 0.0)
            if p_stock > 0.0 and bt(p_stock):
                print(f"  [Stockout] Agent {i} on method {m} → discontinued")
                ppl.method[i] = 0
                ppl.on_contra[i] = False
