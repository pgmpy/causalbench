# algo-benchmarks: Conditional Independence (CI) Tests Benchmark

## Overview

This benchmark framework evaluates the performance of various Conditional Independence (CI) tests, primarily those in [pgmpy](https://github.com/pgmpy/pgmpy). CI tests may perform differently depending on the data-generating mechanism (DGM), sample size, variable types, effect size, and the complexity of the conditioning set. **algo-benchmarks** helps users and developers:

- Compare CI tests under standardized, reproducible settings.
- Select the best CI test for their data and use case.
- Contribute new tests or data-generating mechanisms for further comparison.

Benchmark results are saved to CSV files, which can be used to generate plots or for further analysis.

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Clone this repository (`algo-benchmarks`)
- Install [pgmpy](https://github.com/pgmpy/pgmpy) (either latest release or in editable mode)

### Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

If you want to use a development version of pgmpy:
```bash
git clone https://github.com/pgmpy/pgmpy.git
cd pgmpy
pip install -e .[tests]
```
Return to your `algo-benchmarks` directory before running benchmarks.

---

## Usage

### Running the Benchmark

From the root directory of `algo-benchmarks`, run:
```bash
python -m PY_Scripts.CI_Benchmarks
```

This will:
- Run each CI test on each DGM for various sample sizes, conditioning set sizes, and effect sizes.
- Output detailed and summary CSV files (`ci_benchmark_raw_result.csv`, `ci_benchmark_summaries.csv`).

### Custom Data-Generating Mechanisms

To add your own DGM:
1. Define a function in `PY_Scripts/data_generating_mechanisms.py`:
   ```python
   def my_custom_dgm(n_samples, effect_size=1.0, n_cond_vars=1, seed=None, dependent=True):
       # return a pandas.DataFrame with columns like ['X', 'Y', 'Z1', ...]
   ```
2. Register it in the DGM registry in that file:
   ```python
   DGP_REGISTRY["my_custom"] = my_custom_dgm
   ```
3. Add `"my_custom"` to the `DGM_TO_CITESTS` mapping in `CI_Benchmarks.py` if you want it benchmarked.

---

## Understanding the Output

You get two main files after running the benchmark:

- **`ci_benchmark_raw_result.csv`**: All individual benchmark runs.
- **`ci_benchmark_summaries.csv`**: Aggregated summary statistics.

### Output Columns

#### `ci_benchmark_raw_result.csv`
| Column         | Description                                                        |
|----------------|--------------------------------------------------------------------|
| dgm            | Data Generating Mechanism used                                     |
| sample_size    | Number of samples                                                  |
| n_cond_vars    | Number of conditioning variables                                   |
| effect_size    | Numeric effect size (0 = null, >0 = alt)                           |
| repeat         | Repetition index                                                   |
| ci_test        | CI test used (e.g., pearsonr, gcm, pillai)                        |
| dependent      | `True` if X and Y are dependent, `False` otherwise                 |
| p_value        | The test's p-value                                                 |

#### `ci_benchmark_summaries.csv`
| Column           | Description                                                    |
|------------------|----------------------------------------------------------------|
| dgm              | Data Generating Mechanism used                                 |
| sample_size      | Number of samples                                              |
| n_cond_vars      | Number of conditioning variables                               |
| effect_size      | Effect size                                                    |
| ci_test          | CI test used                                                   |
| significance_level | Significance threshold used                                  |
| type1_error      | False positive rate                                            |
| type2_error      | False negative rate                                            |
| power            | 1 - type2_error                                                |
| N_null           | Number of null runs                                            |
| N_alt            | Number of alt runs                                             |

---

## Customizing the Benchmark

### Adding a New DGM

1. Edit `benchmarks/DGM.py` and define your function.
2. Add it to the `DGP_REGISTRY` dictionary.
3. Optionally, add it to the `DGM_TO_CITESTS` mapping in `ci_benchmarks.py`.

### Adding a New CI Test

1. Implement your test as a function (compatible with pgmpy’s CI test callable signature).
2. Register it in the `ci_tests` dictionary in `CI_Benchmarks.py`.
3. Add it to the list for relevant DGMs in `DGM_TO_CITESTS`.

---

## Plotting and Visualization

You can create plots from the summary CSV using pandas, matplotlib, or seaborn.  
See the plotting functions in `CI_Benchmarks.py` for example usage.

---

## Contribution Guidelines

- Please add tests for any new functionality.
- Follow the code style used in pgmpy and this repo.
- Document any new DGMs or CI tests in this file.

---

## References

- [pgmpy documentation](https://pgmpy.org/)
- [pgmpy/pgmpy#2150](https://github.com/pgmpy/pgmpy/issues/2150)
- Relevant academic papers as needed.