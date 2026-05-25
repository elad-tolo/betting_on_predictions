# Semi-Supervised Hypothesis Testing by Betting on Predictions

Code for the paper
**"Semi-Supervised Hypothesis Testing by Betting on Predictions"**
Yaniv Tenzer*, Elad Tolochinsky*, Yaniv Romano. ICML 2026.

We introduce a testing-by-betting framework that leverages predictions on
unlabeled data to enhance the power of sequential hypothesis testing. Given
limited labeled samples from the joint distribution of `(X, Y)` and additional
unlabeled samples from the marginal of `X`, we construct an *imputed e-statistic*
that bets on model predictions. Under label-shift or concept-shift assumptions
the resulting sequential test is anytime valid in finite samples regardless of
the predictive model's accuracy, and we prove non-trivial power for the binary
case. Empirically the method improves over likelihood-ratio and PPI baselines
even when the unlabeled set is small or `X`/`Y` are weakly correlated.

This repository reproduces the simulations of Sections 5–7 of the paper.

## Repository layout

```
.
├── algs.py                          # Predictive models: NB (Section 4.1), Bayes (4.2), TabPFN (Section 7)
├── simulations.py                   # Runs sequential tests, aggregates rejection rates → power
├── sequential_tests/
│   ├── imputed.py                   # Our imputed e-statistic (Section 3.1, eq. 2)
│   ├── lrt.py                       # Likelihood-ratio baselines: e^Y_LR, e^X_LR, e^{Y|X}_LR
│   └── ppi.py                       # Prediction-Powered Inference baseline (Appendix E)
├── utils/
│   ├── data_sampler.py              # Synthetic + real-data samplers (label / concept shift, healthcare, LLM)
│   ├── ada_grad.py                  # AdaGrad for online tuning of γ in K(·)
│   └── plotting.py                  # Power-vs-step plot helpers
├── mains/
│   ├── main_label_shift.py          # Section 5.4 — synthetic label shift
│   ├── main_concept_shift.py        # Section 5.5 — synthetic concept shift
│   ├── main_label_shift_tune_M.py   # Section F.3 — power as a function of M
│   ├── main_llm_judge_label_shift.py# Section 6.3 — LLM-judge as unlabeled X
│   ├── main_llm_concept_shift.py    # Section 6.2 — dataset identifier as X
│   ├── multivariate_health_care.py  # Section 7  — California census + TabPFN
│   └── *.sh                         # SLURM submission scripts
├── data/                            # All datasets used in the paper (shipped)
│   ├── health_care/                 # features_2017/2018/2019.parquet for Section 7
│   ├── math/                        # (X, Y) parquets for Section 6.2 (LLM concept shift)
│   └── math_judge/                  # (X, Y) parquets for Section 6.3 (LLM-judge label shift)
├── data_generation/                 # Reproducibility only — not required to run experiments
│   ├── math_data_gen.py             # Generate completions with vLLM
│   ├── process_math_generations.py  # Verify completions → (X, Y) parquets in data/math*/
│   └── health_care.ipynb            # Build features_YYYY.parquet from California ACS
└── plot_simulation_results.ipynb    # Reproduce figures from saved results
```

## Setup

The code targets Python 3.11. Conda is recommended because the
`multivariate_health_care` experiment uses `tabpfn` with CUDA.

```bash
conda create -n ml_hypothesis python=3.11
conda activate ml_hypothesis
pip install -r requirements.txt
```

All datasets used in the paper ship with the repository under `data/`, so the
experiments below can be run directly — no external downloads or data
regeneration required.

> Optional — the `data_generation/` scripts are included for transparency and
> let you regenerate the `data/math*/` parquets or the healthcare parquets
> from scratch. Install the additional dependencies with
> `pip install -r requirements-data-gen.txt`. These are *not* required to
> run any of the experiments below.

## Reproducing the experiments

All `main_*` scripts read their config from constants near the top of the file
and write results under `results/`. Run them from the project root using
module form so imports resolve correctly:

| Paper section | Command |
|---|---|
| 5.4 — Label-shift simulation | `python -u -m mains.main_label_shift` |
| 5.5 — Concept-shift simulation | `python -u -m mains.main_concept_shift` |
| F.3 — Power as a function of `M` | `python -u -m mains.main_label_shift_tune_M` |
| 6.2 — LLM concept shift (dataset id as `X`) | `python -u -m mains.main_llm_concept_shift` |
| 6.3 — LLM-judge label shift | `python -u -m mains.main_llm_judge_label_shift` |
| 7   — Multivariate healthcare (TabPFN) | `python -u -m mains.multivariate_health_care` |

The healthcare script also accepts CLI flags
(`--n_test`, `--n_train`, `--seed`, `--steps`, `--M`, `--repetitions`, `--validity`).
To parallelize the 100-seed sweep across GPUs, we ship SLURM templates under
`mains/`: [`run_multivariate_all.sh`](mains/run_multivariate_all.sh) submits
one [`submit_multivariate_sbatch.sh`](mains/submit_multivariate_sbatch.sh) job
per starting seed, and each job runs 10 seeds (`--repetitions 10`) on a single
GPU — so the 100 seeds are split into 10 jobs of 10 seeds each.

### Plotting

After each script writes a `.pkl` under `results/`, use
[`plot_simulation_results.ipynb`](plot_simulation_results.ipynb) to reproduce
the figures in the paper.


## License

See [LICENSE](LICENSE).
