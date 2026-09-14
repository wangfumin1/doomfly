# SNN research baseline

This branch develops a reproducible associative-learning benchmark on the retained
MaleCNS v1.0 graph. It does not treat changed weights, long Doom survival, or a
single favorable episode as evidence that the reconstructed fly learned a task.

The broader virtual-world research direction is documented in
[`snn-virtual-world-research-plan.md`](snn-virtual-world-research-plan.md).

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

## One-command local workflow

The intended host for the current research branch is WSL2/Ubuntu or another
Linux environment with Python 3.11 and `clang++`.

For a fresh machine/run:

```sh
git checkout research/snn-baseline
bash scripts/snn_research_local.sh all first-local
```

`all` performs:

1. Python 3.11 virtual-environment setup;
2. pinned dependency installation;
3. MaleCNS v1.0 download and checksum verification;
4. graph preparation and data audit;
5. native neural/plasticity kernel builds;
6. lightweight protocol/decision tests;
7. prepared-runtime model tests;
8. visual cue preflight;
9. centered-v6 strict conditioning if preflight passes; and
10. historical v5-timing replication on v6.

Every independent experiment needs a new run ID. Existing experiment directories
are never overwritten. The main artifact to return for analysis is:

```text
outputs/doom-learning/research-runs/<run-id>/research-report.json
```

If a stage crashes, keep the partial directory and terminal traceback.

Useful individual commands are:

```sh
bash scripts/snn_research_local.sh bootstrap
bash scripts/snn_research_local.sh prepare
bash scripts/snn_research_local.sh smoke
bash scripts/snn_research_local.sh runtime-tests
bash scripts/snn_research_local.sh baseline my-run
bash scripts/snn_research_local.sh legacy my-run
bash scripts/snn_research_local.sh report my-run
```

## Added research stages

### 1. Lightweight decision-logic tests

These tests require no MaleCNS download and no compiled neural kernel:

```sh
python -m pytest \
  tests/test_doom_learning_v6_conditioning.py \
  tests/test_doom_learning_v6_conditioning_strict.py \
  tests/test_doom_learning_v6_cue_screen.py \
  tests/test_doom_learning_v6_research_report.py -q
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
  --model centered-v6 \
  --preflight outputs/doom-learning/physiology-v6/cue-screen-run1/results.json \
  --out outputs/doom-learning/physiology-v6/conditioning-strict-run1
```

An explicitly requested pair is accepted only if that pair passed preflight.

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
cue variants/replications before making a robust learning claim.

### 5. Preregistered plasticity comparison

Before seeing the new local baseline result, this branch registers one comparison
mechanism: `eligibility-ltd-v6`.

The normal `centered-v6` path disables the older native LTD write and applies the
Python centered rate rule. The comparison model instead enables the already
preserved native mechanism:

- KC spikes create a 1-second eligibility trace;
- actual PPL101 spikes gate depression;
- reconstructed PPL101->MBON11 contact fractions determine edge-specific gain;
- only the same existing KC->MBON11 edges are plastic;
- efficacy has the same 0.1x lower bound;
- v6 sensory encoding, KC dynamics, tonic calibration and strict protocol remain
  unchanged.

It is deliberately an LTD-only candidate, not a final proposed learning model.
Its role is to determine whether explicit spike-local eligibility performs
differently from the centered rate rule under the exact same assay.

If `research-report.json` says `strict_learning_gate_failed`, run:

```sh
bash scripts/snn_research_local.sh phase1 <run-id>
```

This first executes prepared-runtime candidate tests, then writes:

```text
conditioning-strict-eligibility/results.json
```

and regenerates `research-report.json` with the side-by-side outcome.

No eta search is performed automatically. If both registered rules fail, that
negative result is preserved before a new three-factor candidate is designed.

## Next model questions after the benchmark is trustworthy

Do not tune against Doom survival first. The preferred sequence is:

1. establish a balanced and distinguishable visual cue pair;
2. reproduce a cue-specific associative effect under strict controls;
3. compare centered-v6 and the preregistered eligibility-LTD candidate if needed;
4. repeat a passing rule across independent cue variants without retuning;
5. test memory retention, erasure and transfer to held-out visual variants;
6. attribute the effect through circuit ablation/randomized-topology controls; and
7. only then move the selected mechanism into a simpler non-stationary virtual
   world before returning to complex game/NPC environments.

Any alternative learning rule should retain the released graph during this phase
and keep sensory input, neural dynamics, reinforcement, plasticity and motor
decoding separately auditable.
