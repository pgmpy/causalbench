import numpy as np

from pgmpy.base import DAG
from pgmpy.estimators import PC, GES
from pgmpy.metrics import SHD
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN


def generate_random_dag(num_nodes: int, edge_prob: float = 0.3, seed: int = 0) -> DAG:
    dag = DAG.get_random(n_nodes=num_nodes, edge_prob=edge_prob, seed=seed)
    for i in range(num_nodes):
        dag.add_node(f"X_{i}")
    return dag

def compute_shd_direct(true_dag, learned_dag) -> int:
    E_true = set(true_dag.edges())
    E_est  = set(learned_dag.edges())
    return len(E_true.symmetric_difference(E_est))

num_trials = 10
shd_pc_list = []
shd_ges_list = []

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
    
    try:
        learned_dag_pc = PC(data).estimate(
            ci_test="pearsonr",
            variant="stable",
            return_type="dag",
        )
    except Exception as e:
        print(" PC estimation failed:", e)
        continue
    
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

    for g in (learned_dag_pc, learned_dag_ges):
        g.add_nodes_from(true_dag.nodes())

    shd_pc  = compute_shd_direct(true_dag, learned_dag_pc)
    shd_ges = compute_shd_direct(true_dag, learned_dag_ges)

    shd_pc_list.append(shd_pc)
    shd_ges_list.append(shd_ges)

    print(" SHD (PC):", shd_pc)
    print(" SHD (GES):", shd_ges)

print(f"\nAverage SHD over {len(shd_pc_list)} successful trials:")
print(f"  PC:  {np.mean(shd_pc_list):.2f} ± {np.std(shd_pc_list):.2f}")
print(f"  GES: {np.mean(shd_ges_list):.2f} ± {np.std(shd_ges_list):.2f}")
