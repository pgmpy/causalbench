import numpy as np
from pgmpy.base import DAG
from pgmpy.estimators import PC, GES
from pgmpy.metrics import SHD
from pgmpy.factors.continuous import LinearGaussianCPD
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

def generate_random_dag(num_nodes: int, edge_prob: float = 0.3, seed: int = 0) -> DAG:
    dag = DAG.get_random(n_nodes=num_nodes, edge_prob=edge_prob, seed=seed)
    for i in range(num_nodes):
        dag.add_node(f"X_{i}")
    return dag

# Benchmark parameters
num_trials = 10
shd_pc_list = []
shd_ges_list = []

# Run trials
for trial in range(num_trials):
    np.random.seed(trial)
    print(f"\nTrial {trial + 1}/{num_trials}")

    true_dag = generate_random_dag(num_nodes=5, edge_prob=0.3, seed=trial)

    lgbn = LGBN(true_dag.edges())
    lgbn.add_nodes_from(true_dag.nodes())
    for node in true_dag.nodes():
        parents = list(lgbn.get_parents(node))
        beta = [0.0] + list(np.random.uniform(0.5, 1.5, size=len(parents)))
        cpd = LinearGaussianCPD(variable=node, beta=beta, std=1, evidence=parents)
        lgbn.add_cpds(cpd)

    data = lgbn.simulate(n=1000)

    # PC Estimation
    try:
        learned_dag_pc = PC(data).estimate(
            ci_test="pearsonr",
            variant="stable",
            return_type="dag",
        )
    except Exception as e:
        print(" PC estimation failed:", e)
        continue

    # GES Estimation
    try:
        ges_out = GES(data).estimate(scoring_method="bic-g")
        learned_dag_ges = (
            ges_out["model"]
            if isinstance(ges_out, dict) and "model" in ges_out
            else (ges_out[0] if isinstance(ges_out, tuple) else ges_out)
        )
    except Exception as e:
        print(" GES estimation failed:", e)
        continue

    # Ensure node alignment
    all_nodes = sorted(set(true_dag.nodes()).union(
        set(learned_dag_pc.nodes())).union(set(learned_dag_ges.nodes())))
    true_dag.add_nodes_from(all_nodes)
    learned_dag_pc.add_nodes_from(all_nodes)
    learned_dag_ges.add_nodes_from(all_nodes)

    # Compute SHD using built-in method
    try:
        shd_pc = SHD(true_dag, learned_dag_pc)
        shd_ges = SHD(true_dag, learned_dag_ges)
    except Exception as e:
        print(" SHD computation failed:", e)
        print(" true_dag edges:", true_dag.edges())
        print(" learned_dag_pc edges:", learned_dag_pc.edges())
        print(" learned_dag_ges edges:", learned_dag_ges.edges())
        continue

    shd_pc_list.append(shd_pc)
    shd_ges_list.append(shd_ges)

    print(" SHD (PC):", shd_pc)
    print(" SHD (GES):", shd_ges)

# Final Results
print(f"\nAverage SHD over {len(shd_pc_list)} successful trials:")
print(f"  PC:  {np.mean(shd_pc_list):.2f} ± {np.std(shd_pc_list):.2f}")
print(f"  GES: {np.mean(shd_ges_list):.2f} ± {np.std(shd_ges_list):.2f}")
