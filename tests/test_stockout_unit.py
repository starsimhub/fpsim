# python -m tests.test_stockout_unit
# tests/test_stockout_unit.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fpsim as fp
from fpsim.stockout import StockoutIntervention
from fpsim.methods import make_methods


# --- Helper to save agent-level data ---
def save_agent_data(sim, filename, year=None):
    people = sim.people
    df = pd.DataFrame({
        'uid': people.uid,
        'age': people.age,
        'alive': people.alive,
        'sex': people.sex,
        'method': people.method,
        'pregnant': people.pregnant,
        'postpartum': people.postpartum,
    })

    # Filter to women of reproductive age
    df = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]

    # Add method names
    method_defs = make_methods()
    inv_method_map = {v: k for k, v in method_defs.method_map.items()}
    df['method_name'] = df['method'].map(inv_method_map)

    if year:
        df['year'] = year

    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


# --- Define stockout intervention ---
stockout_probs = {year: {1: 1.0, 3: 1.0} for year in range(2025, 2031)}
stockout = StockoutIntervention(stockout_probs=stockout_probs, seed=42)

# --- Run sim without stockout ---
sim_base = fp.Sim(location='kenya', start_year=2020, end_year=2030, n_agents=1000, label='baseline')
sim_base.run()
save_agent_data(sim_base, 'baseline_2030_agents.csv', year=2030)

# --- Run sim with stockout ---
sim_stock = fp.Sim(location='kenya', start_year=2020, end_year=2030, n_agents=1000, label='stockout')
sim_stock['interventions'] = [stockout]
sim_stock.run()
save_agent_data(sim_stock, 'stockout_2030_agents.csv', year=2030)

# --- Print method usage for method 1 and 3 from 2025 to 2030 ---
print("\n[DEBUG] Annual method usage for methods 1 and 3 (2025–2030):")
for year_offset, method_list in enumerate(sim_stock.results['method_usage']):
    year = sim_stock['start_year'] + year_offset
    if 2025 <= year <= 2030:
        m1 = method_list[1] * 100
        m3 = method_list[3] * 100
        print(f"Year {year}: Method 1 = {m1:.1f}%, Method 3 = {m3:.1f}%")


# --- CPR plot comparison ---
plt.figure(figsize=(10, 5))
plt.plot(sim_base.results['t'], sim_base.results['cpr'] * 100, label='No stockout')
plt.plot(sim_stock.results['t'], sim_stock.results['cpr'] * 100, label='With 100% stockout (Pill & Injectable)', linestyle='--')
plt.axvspan(2025, 2030, color='gray', alpha=0.2, label='Stockout period')
plt.xlabel('Year')
plt.ylabel('CPR (%)')
plt.title('Kenya: Impact of Method 1 (Pill) & 3 (Injectable) Stockout on CPR')
plt.ylim(0, 60)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
