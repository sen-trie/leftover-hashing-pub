# leftover-hashing-pub

Computes and compares two quantum leftover hashing bounds for the phase-error state:

- the exact smooth min-entropy bounds of Regula and Tomamichel, achievability `ell_low` and converse `ell_up`, from a closed-form solution of the smooth min-entropy SDP
- the sandwiched Rényi bound `ell_alpha` of Dupuis

The file also includes the original SDP, as a check on the closed form for small `n`.

## Requirements

numpy, scipy, cvxpy, toqito

## Usage

    python leftover_hashing.py

This checks the SDP against the closed form for n = 1–3, then prints `H_low`, `H_up`, `ell_low`, `ell_up`, `ell_alpha` and the optimal `alpha` for n = 100, 200, 500, 1000. The results are saved to `key_lengths.csv`.

The defaults are `P_ERR = 0.01`, `EPS = 1e-6` and `MU = EPS / 2`. You can change them, along with `N_TABLE`, at the top of the file.
