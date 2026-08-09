# Behavioral Economics Decision Intelligence Platform

An end-to-end machine-learning and behavioural-economics research pipeline built on **real, publicly archived experimental decision-making data**. Everything — dataset discovery, download, cleaning, modelling, causal analysis, explainability, optimisation and reporting — runs from a single notebook with no manual data handling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Headline result:** on 14,568 archived risky-choice problems, gradient boosting cuts test error **61%** below a properly fitted Cumulative Prospect Theory model (MSE 0.0096 vs 0.0246). On 493,755 intertemporal-choice trials from 100 independent studies, ML reaches **0.909 ROC-AUC** against 0.635 log loss for quasi-hyperbolic discounting — a **40%** log-loss improvement. Feeding the behavioural model's own prediction into the ML model as a feature improves it further.

---

## What this is

Two modelling components plus causal, explanatory and optimisation layers:

**Component A — Behavioural Pattern Prediction Engine.** Predicts experimentally observed choice behaviour from the decision environment (rewards, delays, probabilities, outcome distributions) and a participant's own previously observed choices. Also estimates *continuous* behavioural parameters — discount rates, present bias, loss aversion, probability weighting — rather than forcing everything into binary labels.

**Component B — Behavioural Intervention Optimiser.** Estimates `P(choice | environment, history)` and searches the permissible space of decision environments for settings that optimise an explicitly defined objective under cost constraints.

### Scope statement

The models predict **experimentally observed behavioural decision patterns**. They do not diagnose psychological conditions and do not classify personality. An estimated `β` of 0.76 means *this person took the sooner option more often than a time-consistent model predicts, in this task, on this day* — not that they are impulsive. Discount rates move with mood, sleep, stress and framing.

---

## Datasets

All three are downloaded automatically at runtime. Nothing is bundled, uploaded, or synthetic.

| Track | Dataset | Scale | Phenomenon | Source |
|---|---|---|---|---|
| `RISK` | **choices13k** — Peterson et al., *Science* 2021 | 14,568 problems | risk preference, loss aversion, probability weighting | [GitHub](https://github.com/jcpeterson/choices13k) |
| `ITC` | **ITC Database** — Pongratz & Schoemann, *Scientific Data* 2026 | 1,172,644 trials · 11,852 participants · 100 studies | intertemporal choice, delay discounting, present bias | [OSF](https://osf.io/3wsae/) |
| `RISK_IND` | **CPC18 calibration set** — Plonsky, Erev & Ert | 510,750 decisions · 686 participants | risky choice, ambiguity, description–experience gap | [Zenodo](https://zenodo.org/records/845873) |

The notebook catalogues **12 candidates and documents all 9 rejections**, including ICPSR (needs credentials), Many Labs 1 (no repeated trials, no continuous intervention variables) and generic Kaggle marketing CSVs (no decision data at all). Selection happened *before* modelling.

**Licences:** choices13k is an academic release accompanying a *Science* paper; CPC18 is CC BY 4.0; **the ITC Database is CC BY-NC-SA 4.0 — non-commercial use only.**

---

## Results from the reference run

### ML vs classical behavioural theory

| Track | Best classical model | Best ML model | Improvement |
|---|---|---|---|
| RISK | Cumulative Prospect Theory — MSE 0.0246 | XGBoost + CPT feature — MSE 0.0096 | **61.1%** |
| ITC | Quasi-hyperbolic (β–δ) — log loss 0.635 | Tuned HistGradientBoosting — log loss 0.378 | **40.5%** |

The theory ladder orders exactly as theory predicts: Expected Value (0.0315) → CRRA utility (0.0279) → CPT (0.0246), against a mean-predictor baseline of 0.0508.

### Estimated behavioural parameters

Fitted by maximum likelihood on exact published outcome distributions:

- **Cumulative Prospect Theory:** α = 0.824 (gain curvature), β = 0.709 (loss curvature), **λ = 1.371 (loss aversion)**, γ_gain = 0.793, γ_loss = 0.775 — inverse-S probability weighting, all in the direction the literature reports.
- **Per-participant discounting:** thousands of individual β–δ fits, with participants whose estimates rest on a search bound reported as **not identified** rather than counted as measurements.

### Causal finding

The one causal claim in the project comes from a genuine randomised contrast: in choices13k, **1,562 problems appear both with and without outcome feedback**, holding the gamble fixed.

**ATE of experience vs description = +0.0149** on choice rate, 95% bootstrap CI [+0.0082, +0.0218], paired *t* = 4.33 (p = 1.6e-5), Wilcoxon p = 2.7e-4, Cohen's d_z = 0.11. IPW and doubly-robust estimators are reported as sensitivity checks.

**A negative result, reported as found:** theory predicts this gap should concentrate in rare-event problems. It did not — the effect was larger for common-event problems and the difference was not significant (p = 0.074). The notebook prints an explicit verdict rather than assuming the textbook outcome.

### Generalisation

- **Leave-one-study-out** across 12 held-out studies: out-of-domain AUC **0.891** vs theory's 0.732.
- **choices13k → CPC18** (different lab, country, incentives): AUC drops **0.912 → 0.751**, a 17.7% degradation. This is the honest cost of dataset shift and matches what Thomas et al. (2024, *Nature Human Behaviour*) report for models trained on choices13k.

### Most predictive features (SHAP)

For intertemporal choice the dominant driver is **the participant's own previously observed patience** (|SHAP| = 1.36), well ahead of any property of the offer itself. For risky choice it is the **CPT model's own prediction**, then the expected-value advantage — theory-derived quantities carry most of the signal.

---

## How to run

1. Upload `Behavioral_Economics_Decision_Intelligence_Platform.ipynb` to [Google Colab](https://colab.research.google.com/).
2. **Runtime ▸ Run all.**

No downloads, uploads, API keys or URL edits. Google Drive mounting is optional and the notebook works without it.

**Runtime:** roughly 45–70 minutes on a free Colab CPU instance. For a ~10 minute pass, set `CONFIG.quick_mode = True` in the configuration cell.

### Configuration

```python
CONFIG = Config(
    random_seed   = 42,
    test_size     = 0.15,
    val_size      = 0.15,
    quick_mode    = False,      # True => small trial counts, fast pass
    max_itc_rows  = 400_000,    # None => all 1.17M trials (slower)
    optuna_trials = 30,
    launch_gradio = True,
)
```

The intertemporal track is capped at 400,000 trials by default, subsampled **by participant** so nobody is partially observed.

---

## Methods

**Behavioural economics.** Expected value; CRRA/power expected utility; Cumulative Prospect Theory with rank-dependent probability weighting; exponential, Mazur hyperbolic and quasi-hyperbolic (β–δ) discounting. All fitted by maximum likelihood with multi-start optimisation and **explicit boundary-solution detection** — a parameter resting on a search bound is reported as not identified, never as a finding.

> The CPT rank ordering depends only on outcome values, never on fitted parameters, so ranking and cumulative probabilities are precomputed once into a "prospect tableau". This turns a ~10⁴ × 10² Python loop into a handful of array operations and makes full-dataset MLE fitting take seconds instead of hours.

**Machine learning.** Logistic/linear baselines, Ridge/Lasso, Random Forest, HistGradientBoosting, XGBoost, LightGBM, MLP. Optuna tunes against the validation split only; the test split is touched once.

**Hybrid models.** Every ML model is also fitted with the classical model's prediction as an extra feature, testing whether theory and ML are complementary (the "cognitive model prior" idea from Bourgin et al., 2019). They are.

**Validation.** Participant-grouped and problem-grouped 70/15/15 splitting — never row-random. In the ITC track a participant contributes hundreds of trials; scattering them across splits would let the model memorise that individual's baseline patience.

**Causal inference.** Paired within-problem randomised contrast, bootstrap CIs, Wilcoxon signed-rank, causal-forest-style CATE, IPW and AIPW doubly-robust sensitivity checks.

**Explainability.** SHAP global/local attributions with a behavioural-economics glossary, plus model-predicted counterfactuals explicitly distinguished from causal effects.

---

## Notebook structure

```
 1–3   Overview, research questions, behavioural economics background
 4–5   Environment setup, configuration, directories, logging
 6–7   Dataset discovery (12 candidates) and selection rationale
 8     Automated ingestion — retries, fallbacks, caching, Drive mirroring
 9     Data validation and the cleaning decision log
10–11  EDA and the unified behavioural schema
12–13  Behavioural feature engineering and target construction
14–15  Grouped train/val/test splitting and the leakage audit
16     Behavioural economics baselines (the theory ladder)
17–19  ML baselines, advanced models, Optuna tuning
20     Evaluation and the headline theory-vs-ML comparison
21–22  SHAP explainability and per-participant parameter estimation
23–25  Causal inference, intervention analysis, counterfactuals
26–27  Intervention optimiser and scenario simulator
28–29  Cross-dataset generalisation and robustness
30–31  Ethics/fairness audit and final model selection
32–33  Gradio application and automated reporting
34–35  Executive summary, conclusions, limitations, future work
```

## Generated outputs

```
outputs/     dataset_catalog.csv · data_quality_report.csv · cleaning_decisions.csv
             leakage_audit.csv · model_comparison.csv · behavioral_model_results.csv
             behavioral_parameters.csv · causal_effects.csv · intervention_analysis.csv
             optimization_results.csv · cross_dataset_generalization.csv
             robustness_analysis.csv · fairness_subgroup_analysis.csv
             scenario_simulation.csv · shap_importance_*.csv · test_predictions.csv
             experiment_tracking.csv · ingestion_log.csv
figures/     24 publication-quality PNGs
reports/     behavioral_economics_ml_report.html (self-contained, figures embedded)
             executive_summary.txt
models/      final_model_ITC.joblib · final_model_RISK.joblib
```

## Gradio application

Five tabs, launched with a public share link at the end of the notebook: behavioural prediction (intertemporal and risky choice), intervention simulator, constrained optimiser, SHAP explainability, and a datasets/research summary.

---

## Honest scope reductions

Three places where the data did not support what a project like this usually claims. Each is a deliberate choice, documented in the notebook rather than papered over.

1. **No pricing optimiser.** No archived, credential-free, row-level retail pricing or anchoring experiment exists. Rather than fabricate prices, the module optimises real decision variables — reward magnitudes and delays — under an explicit expected-cost objective. The optimiser includes a policy-fixed **minimum waiting period**, because without one it finds the trivial solution of collapsing the delay to a single day.
2. **No anchoring, framing or default labels.** These datasets contain no reference price, gain/loss frame manipulation or default option, so those targets are not constructed. Inventing them would be fabrication.
3. **One causal claim only.** Study-level moderators in the ITC Database (country, incentivisation, online vs lab) are confounded with study. They are reported as association, flagged as such in the intervention table.

## Limitations

- Predicts observed choice behaviour, **not** psychological states or diagnoses.
- Estimated parameters are protocol- and context-specific, not stable traits.
- Only the description–experience contrast is causally identified, and it is small (≈1.5 pp).
- Optimiser output is a model-based recommendation requiring randomised validation.
- Participant pools (undergraduates, MTurk workers) are not population-representative.
- Not for consequential individual decisions — credit, insurance, hiring, clinical judgement. The cross-dataset results show performance degrades across contexts.
- Choice architecture is not ethically neutral. The same machinery that finds the cheapest way to encourage saving finds the cheapest way to encourage borrowing; that distinction lives in governance, not in the loss function.

## Citations

If you use this pipeline, cite the underlying data:

```bibtex
@article{peterson2021choices13k,
  title={Using large-scale experiments and machine learning to discover
         theories of human decision-making},
  author={Peterson, Joshua C. and Bourgin, David D. and Agrawal, Mayank and
          Reichman, Daniel and Griffiths, Thomas L.},
  journal={Science}, volume={372}, number={6547}, pages={1209--1214}, year={2021},
  doi={10.1126/science.abe2629}
}

@article{pongratz2026itc,
  title={A database of choice and response time data in intertemporal choice},
  author={Pongratz, Ludwig and Schoemann, Martin},
  journal={Scientific Data}, volume={13}, pages={323}, year={2026},
  doi={10.1038/s41597-026-06947-4}
}

@article{erev2017cpc,
  title={From anomalies to forecasts: Toward a descriptive model of decisions
         under risk, under ambiguity, and from experience},
  author={Erev, Ido and Ert, Eyal and Plonsky, Ori and Cohen, Doron and Cohen, Oded},
  journal={Psychological Review}, volume={124}, number={4}, pages={369--409}, year={2017},
  doi={10.1037/rev0000062}
}
```

## Licence

Pipeline code: MIT. **Data carries its own terms — the ITC Database is CC BY-NC-SA 4.0, so derived work using that track is non-commercial.**
