import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.base import DAG
from pgmpy.estimators import PC, GES
from pgmpy.metrics import SHD
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

"""
Benchmarking Structural Hamming Distance (SHD) for Causal Discovery Algorithms: PC and GES

Algorithm Definitions:
----------------------
- PC (Peter-Clark) Algorithm:
  A constraint-based algorithm that starts with a complete undirected graph and removes edges
  based on conditional independence tests. It then orients edges using separation sets and
  rules like the collider rule:
      X → Z ← Y   if X ⟂⟂ Y | Z

- GES (Greedy Equivalence Search) Algorithm:
  A score-based algorithm that performs greedy forward and backward search in the space
  of equivalence classes of DAGs to maximize a scoring criterion such as BIC.

  Scoring function (Bayesian Information Criterion - BIC):
      Score(G : D) = log P(D | G) - λ * |G|

Metric:
-------
- SHD (Structural Hamming Distance):
  Measures the number of edge insertions, deletions, or reversals required to convert
  one DAG into another.
"""

# Benchmark parameters
num_trials = 10
results = []

for trial in range(num_trials):
    np.random.seed(trial)
    print(f"\nTrial {trial + 1}/{num_trials}")

    # Generate random true model
    lgbn = LGBN.get_random(n_nodes=5, edge_prob=0.3, seed=trial)
    true_dag = DAG(lgbn.edges())
    true_dag.add_nodes_from(lgbn.nodes()) 

    # Simulate data
    data = lgbn.simulate(n=1000)

    # PC estimation
    pc_est = PC(data).estimate(ci_test="pearsonr", variant="stable", return_type="dag")
    learned_dag_pc = nx.DiGraph(pc_est.edges())

    # GES estimation
    ges_out = GES(data).estimate(scoring_method="bic-g")
    ges_model = ges_out["model"] if isinstance(ges_out, dict) else ges_out
    learned_dag_ges = nx.DiGraph(ges_model.edges())

    # Debug: show node sets before SHD
    print(" True nodes:   ", set(true_dag.nodes()))
    print(" PC nodes:     ", set(learned_dag_pc.nodes()))
    print(" GES nodes:    ", set(learned_dag_ges.nodes()))

    # Compute SHD (this will crash if node sets differ)
    shd_pc = SHD(true_dag, learned_dag_pc)
    shd_ges = SHD(true_dag, learned_dag_ges)

    results.append({"trial": trial + 1, "shd_pc": shd_pc, "shd_ges": shd_ges})
    print(" SHD (PC):", shd_pc)
    print(" SHD (GES):", shd_ges)

# Summary
df = pd.DataFrame(results)
print("\nAverage SHD over trials:")
print(f"  PC:  {df['shd_pc'].mean():.2f} ± {df['shd_pc'].std():.2f}")
print(f"  GES: {df['shd_ges'].mean():.2f} ± {df['shd_ges'].std():.2f}")