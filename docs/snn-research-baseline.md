# SNN research baseline

This branch develops a reproducible associative-learning benchmark on the retained
MaleCNS v1.0 graph. It does not treat changed weights, long Doom survival, or a
single favorable episode as evidence that the reconstructed fly learned a task.

## Why the previous conditioning assay is insufficient

The preserved v5 conditioning result failed its development gate. More
importantly, the raw controls identify a major cue imbalance:

- `left_blue` training produced about 925k KC spikes in the no-US/frozen runs.
- `right_blue` training produced only 172 KC spikes in the corresponding runs.
- For cue 0, paired and no-imposed-US produced exactly the same reported
  selectivity (`0.8891389752823702`).
- The two counterbalances therefore did not provide a symmetric test of
  dopamine-dependent association.

Source: `outputs/doom-learning/physiology-v5/conditioning/results.json`.

The first research objective is consequently not to tune eta until Doom improves.
It is to establish a cue pair and protocol for which a conditioning result is
interpretable.

## Added research stages

### 1. Lightweight decision-logic tests

These tests require no MaleCNS download and no compiled neural kernel:

```sh
python -m pytest \
  tests/test_doom_learning_v6_conditioning.py \
  tests/test_doom_learning_v6_conditioning_strict.py \
  tests/test_doom_learning_v6_cue_screen.py -q
```

The matching GitHub Actions workflow is `.github/workflows/snn-research-smoke.yml`.
Forks may require Actions to be enabled once in the GitHub UI before push events
produce runs.

### 2. Visual cue preflight

Prepare the MaleCNS graph and native kernels as described in the repository
README, then run:

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.cue_screen \
  --out outputs/doom-learning/physiology-v6/cue-screen-run1
```

The screen probes `blue`, `green`, `white`, `left_blue`, `right_blue`, `vertical`
and `horizontal` after the same dark warmup used by the conditioning readout. A
candidate pair is eligible only when:

- both KC response patterns differ from the black control;
- the two KC response patterns differ from each other;
- both cues produce a nonzero MBON readout; and
- total KC activity differs by no more than 2x by default.

`results.json` records every pair and a `recommended_pair`. Passing this screen is
only an engineering prerequisite; it is not biological validation of the visual
model.

### 3. Exact v5-protocol replication on v6

To isolate model revision from assay revision, `doom_learning_v6.conditioning`
ports the existing v5 timing and decision gate to the v6 model:

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.conditioning \
  --out outputs/doom-learning/physiology-v6/conditioning-replication-run1
```

This result answers a narrow question: does v6 change the outcome under the old
assay? It should not be preferred over the strict assay for new claims because the
legacy protocol has unequal condition durations and uses the historically
problematic fixed cue pair.

### 4. Preflight-gated strict differential conditioning

The strict experiment requires a completed cue screen. With no explicit cue
arguments, it automatically uses the screen's best eligible pair:

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.conditioning_strict \
  --preflight outputs/doom-learning/physiology-v6/cue-screen-run1/results.json \
  --out outputs/doom-learning/physiology-v6/conditioning-strict-run1
```

An explicitly requested pair is accepted only if that pair passed preflight:

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.conditioning_strict \
  --preflight outputs/doom-learning/physiology-v6/cue-screen-run1/results.json \
  --cue-a vertical --cue-b horizontal \
  --out outputs/doom-learning/physiology-v6/conditioning-strict-vh-run1
```

The strict protocol uses eight 2-second training trials per condition. Each cue is
shown four times in every condition. The four conditions are:

- `paired`: four equal dopamine pulses overlap the four CS+ trials;
- `unpaired`: the same total dopamine dose occurs only outside cue windows and is
  distributed across trials containing both cue identities;
- `frozen`: identical sensory/reinforcement schedule to paired, but weights cannot
  change; and
- `no_imposed_US`: identical visual exposure with no artificial dopamine pulse.

Training duration, cue exposure, and imposed-US dose are checked by pure protocol
logic before a full-graph run starts. After training, weights are frozen during
both cue probes. A memory reset must restore the pre-training response.

The development gate requires both cue counterbalances to show a positive paired
selectivity that exceeds unpaired and no-US controls, while frozen and reset
controls remain exact. Even a passing development gate still requires independent
replicates and a separate held-out behavioral validation before making a learning
claim.

## Next model questions after the benchmark is trustworthy

Do not tune against Doom survival first. The preferred sequence is:

1. establish a balanced and distinguishable visual cue pair;
2. reproduce a cue-specific associative effect under strict controls;
3. repeat across independent cue sets and parameter-preserving starts;
4. compare the current centered anti-Hebbian rule against an explicit
   reward-modulated eligibility/STDP rule on the same 4,184 KC->MBON11 edges;
5. test memory retention, erasure, and transfer to held-out visual variants; and
6. only then ask whether the learned state improves a closed-loop task.

Any alternative learning rule should retain the released graph and keep sensory
input, neural dynamics, reinforcement, plasticity and motor decoding separately
auditable.
