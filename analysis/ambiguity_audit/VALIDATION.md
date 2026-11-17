# Validation — 2026-11-17

- 6 unit tests passed with NumPy 2.3.5 / SciPy 1.17.0.
- End-to-end CLI smoke test passed on synthetic data: 24 clips, 8 actors, two sources, 100 bootstrap draws; audit JSON and four CSV outputs parsed successfully.
- Tests cover exact distribution equality with zero probabilities, distribution error despite matching top labels, tied labels, identical-model paired bootstrap, missing prediction coverage, duplicate rows and inconsistent actor folds.
- No real project predictions were available. No real-data metrics, repository integration, commit verification or GPU runs were performed.
- Script is a standalone export consumer. Source repository adapter and the existing rho_within evaluator remain to be connected.
- Synthetic outputs are intentionally excluded from this package to avoid confusion with research results.
