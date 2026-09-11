# Evaluation Metrics

## Space-Group Conditioning

Detect every generated structure's space group with `spglib` and compare it to
the target stored in the prediction bundle metadata:

```bash
python -m eval.spacegroup --predictions-dir /path/to/prediction/output
```

The default tolerance is `symprec=0.1`. Results are written under the
prediction directory's `eval/` folder as `spacegroup_per_sample.jsonl` and
`spacegroup_summary.json`. The headline per-sample rate includes detection
failures as non-matches, and a separate per-target rate reports whether any
generated sample matched each source target.

## OXtal Metrics

`eval.oxtal` implements the official OXtal evaluation logic. A generated sample
is identified by a CSD refcode, and the evaluator compares that sample to CSD
database truth crystals rather than to the validation tensor target stored in a
Packora batch.

Let $c = 1,\ldots,C$ index CSD refcodes and let $s=1,\ldots,n_c$ index generated
samples for refcode $c$. The number of evaluated samples is:

```math
N = \sum_{c=1}^{C} n_c .
```

### Truth Sets

Official OXtal reads `rigid.csv` and `flexible.csv` under
`$PACKORA_DATA_ROOT/csv_manifests/`, downloaded from `nayoung10/Packora-data`.
Their `truth_refcodes` column contains semicolon-separated polymorph groups. Every listed member
maps to the same truth set. For example:

```text
GLYCIN33;GLYCIN71;GLYCIN98
```

defines:

```text
GLYCIN33 -> {GLYCIN33, GLYCIN71, GLYCIN98}
GLYCIN71 -> {GLYCIN33, GLYCIN71, GLYCIN98}
GLYCIN98 -> {GLYCIN33, GLYCIN71, GLYCIN98}
```

If a validation refcode is absent from the truth map, official OXtal uses the
singleton truth set:

```math
T(c)=\{c\}.
```

For generated sample $x_{c,s}$ and every truth candidate $t\in T(c)$, OXtal
computes COMPACK results at packing sizes 1 and 15. It selects one target
$t^\star_{c,s}$ using only the largest requested packing size, $K=15$:

1. Compute $n^{\mathrm{OX}}_K(x_{c,s},t)$ and
   $\mathrm{rmsd}^{\mathrm{OX}}_K(x_{c,s},t)$ for each $t\in T(c)$.
2. Mark $t$ as passing when
   $n^{\mathrm{OX}}_K(x_{c,s},t)\ge\lceil K\cdot0.5\rceil$.
3. If any target passes, choose the passing target with the smallest
   $\mathrm{rmsd}^{\mathrm{OX}}_K$.
4. If no target passes, choose the target with the largest
   $n^{\mathrm{OX}}_K$.
5. The selected target is $t^\star_{c,s}$. The reported
   $\mathrm{rmsd}_1$, $\mathrm{rmsd}_{15}$, $n_1$, and $n_{15}$ all come from
   this same selected target.

### Per-Sample Indicators

The evaluator writes official OXtal row fields: `clash`, `passed`,
`nmatched_1`, `rmsd_1`, `nmatched_15`, `rmsd_15`, `best_true_refcode`, and
`errors`.

The clash indicator is:

```math
\mathrm{col}_{c,s}
= \mathbf{1}\left[
\exists \text{ heavy-atom nonbonded contact } (i,j):
r_i^\mathrm{vdW}+r_j^\mathrm{vdW}-d_{ij}^{(c,s)} \ge 0.7
\right].
```

The packing indicator is:

```math
\mathrm{pac}_{c,s}
= \mathbf{1}\left[
n^{\mathrm{OX}}_{15}(x_{c,s},t^\star_{c,s}) \ge 8
\right].
```

The recovery indicator includes the official no-clash gate:

```math
\mathrm{rec}_{c,s}
= \mathbf{1}\left[
\mathrm{rmsd}^{\mathrm{OX}}_1(x_{c,s},t^\star_{c,s}) < 0.5
\land
\mathrm{col}_{c,s}=0
\right].
```

Packora reports `Sol_C` for the official OXtal `match_rate` metric. Its
per-sample solve/match indicator is:

```math
\mathrm{sol}_{c,s}
= \mathbf{1}\left[
\mathrm{col}_{c,s}=0
\land \mathrm{pac}_{c,s}=1
\land \mathrm{rmsd}^{\mathrm{OX}}_{15}(x_{c,s},t^\star_{c,s})<2.0
\right].
```

### Reported Metrics

The public Packora summary names are preserved:

```math
\mathrm{Col}_S
= \frac{1}{N}\sum_{c=1}^{C}\sum_{s=1}^{n_c}\mathrm{col}_{c,s}.
```

```math
\mathrm{Pac}_S
= \frac{1}{N}\sum_{c=1}^{C}\sum_{s=1}^{n_c}\mathrm{pac}_{c,s},
\qquad
\mathrm{Pac}_C
= \frac{1}{C}\sum_{c=1}^{C}\max_{1\le s\le n_c}\mathrm{pac}_{c,s}.
```

```math
\mathrm{Rec}_S
= \frac{1}{N}\sum_{c=1}^{C}\sum_{s=1}^{n_c}\mathrm{rec}_{c,s},
\qquad
\mathrm{Rec}_C
= \frac{1}{C}\sum_{c=1}^{C}\max_{1\le s\le n_c}\mathrm{rec}_{c,s}.
```

```math
\widetilde{\mathrm{Sol}}_C
= \frac{1}{C}\sum_{c=1}^{C}\max_{1\le s\le n_c}\mathrm{sol}_{c,s}.
```

`Sol_C` is therefore Packora's retained name for official OXtal
`match_rate`.

### Python Usage

`eval.oxtal.evaluate` requires one CSD refcode per target group:

```python
from ccdc.io import CrystalReader
from eval.oxtal import evaluate

reader = CrystalReader("csd")

target_crystals = [reader.crystal("FETSEC"), reader.crystal("ZERLUD")]
samples_by_target = [
    [reader.crystal("FETSEC"), reader.crystal("ZERLUD")],
    [reader.crystal("ZERLUD"), reader.crystal("FETSEC")],
]
target_refcodes = ["FETSEC", "ZERLUD"]

result = evaluate(target_crystals, samples_by_target, target_refcodes=target_refcodes)
```

The output has sample-level official OXtal fields, target-level OR diagnostics,
and public aggregate metric names:

```python
{
    "summary": {
        "Col_S": 0.0,
        "Pac_S": 0.5,
        "Pac_C": 1.0,
        "Rec_S": 0.5,
        "Rec_C": 1.0,
        "Sol_C": 1.0,
    },
    "per_sample": [
        {
            "target_index": 0,
            "target_identifier": "FETSEC",
            "csd_refcode": "FETSEC",
            "sample_index": 0,
            "best_true_refcode": "FETSEC",
            "clash": False,
            "passed": True,
            "nmatched_1": 1,
            "rmsd_1": 0.0,
            "nmatched_15": 15,
            "rmsd_15": 0.0,
            "recovered": True,
            "matched": True,
            "errors": "",
        },
    ],
    "per_target": [
        {
            "target_index": 0,
            "target_identifier": "FETSEC",
            "csd_refcode": "FETSEC",
            "num_samples": 2,
            "passed": True,
            "recovered": True,
            "matched": True,
        },
    ],
}
```

### PackingSimilarity Settings

OXtal uses CSD COMPACK through `ccdc.crystal.PackingSimilarity`. For packing
size 1, the implementation uses:

| Setting | Value | Purpose |
|---------|-------|---------|
| `packing_shell_size` | `1` | Build the one-molecule shell for `rmsd_1`. |
| `distance_tolerance` | `0.2` | Official size-1 distance tolerance. |
| `angle_tolerance` | `20.0` | Official size-1 angle tolerance. |
| `match_entire_packing_shell` | `True` | Require the whole one-molecule shell to match. |

For packing size 15, the implementation uses:

| Setting | Value | Purpose |
|---------|-------|---------|
| `packing_shell_size` | `15` | Build the 15-molecule shell for `passed` and `rmsd_15`. |
| `distance_tolerance` | `0.5` | Official size-15 distance tolerance. |
| `angle_tolerance` | `75.0` | Official size-15 angle tolerance. |
| `match_entire_packing_shell` | `False` | Permit partial shell matching; OXtal accepts at least 8 of 15 molecules. |

Both shell sizes use:

| Setting | Value |
|---------|-------|
| `timeout_ms` | `10000` |
| `allow_molecular_differences` | `False` |
| `ignore_hydrogen_counts` | `True` |
| `ignore_hydrogen_positions` | `True` | Ignore hydrogen positions in the packing comparison. |
| `ignore_bond_counts` | `True` |
| `ignore_bond_types` | `True` |
| `allow_artificial_inversion` | `True` |

## StructureMatcher Surrogate Metrics

`eval.oxtal_surrogates` implements cheap surrogate metrics for the
same target-grouped setup. These metrics use pymatgen `Structure` inputs and are
intended as fast approximations, not replacements for the CCDC COMPACK-based
OXtal metrics above.

For target structure `x*_c` and generated sample `x_{c,s}`, define:

```math
\mathrm{match}_{c,s}
= \mathbf{1}\left[
\mathrm{StructureMatcher}(x_{c,s}, x_c^\star) \text{ fit is True}
\right].
```

The implementation uses all atoms and:

| StructureMatcher setting | Value |
|--------------------------|-------|
| `stol` | `0.5` |
| `ltol` | `0.3` |
| `angle_tol` | `10.0` |
| `primitive_cell` | `False` |

The local `clash_rate(sample, alpha=0.75)` function returns the fraction of atoms
that participate in at least one short contact under the existing Packora
clash convention. The surrogate collision indicator is:

```math
\mathrm{col}_{c,s}
= \mathbf{1}\left[\mathrm{clash\_rate}(x_{c,s}) > 0\right].
```

This `col_{c,s}` is an all-atom clash surrogate. It is not OXtal's CCDC
intermolecular, heavy-atom, vdW collision check.

The surrogate solve indicator is:

```math
\mathrm{sol}_{c,s}
= \mathbf{1}\left[
\mathrm{col}_{c,s}=0
\land \mathrm{match}_{c,s}=1
\right].
```

The reported metrics are:

```math
\mathrm{SMatch}_S
= \frac{1}{N}\sum_{c=1}^{C}\sum_{s=1}^{n_c}\mathrm{match}_{c,s},
\qquad
\mathrm{SMatch}_C
= \frac{1}{C}\sum_{c=1}^{C}\max_{1\le s\le n_c}\mathrm{match}_{c,s}.
```

```math
\mathrm{SClash}_S
= \frac{1}{N}\sum_{c=1}^{C}\sum_{s=1}^{n_c}
\mathrm{clash\_rate}(x_{c,s}).
```

```math
\mathrm{SSol}_C
= \frac{1}{C}\sum_{c=1}^{C}\max_{1\le s\le n_c}\mathrm{sol}_{c,s}.
```

`eval.oxtal_surrogates.evaluate` returns sample-level diagnostics,
target-level OR diagnostics, and the aggregate summary:

```python
from eval.oxtal_surrogates import evaluate

target_structures = [target_a, target_b]
samples_by_target = [
    [sample_a_0, sample_a_1],
    [sample_b_0, sample_b_1],
]

result = evaluate(target_structures, samples_by_target)
```

The output summary contains:

```python
{
    "SMatch_S": 0.5,
    "SMatch_C": 1.0,
    "SClash_S": 0.0,
    "SSol_C": 1.0,
}
```
