# Virtual-world adaptive SNN research plan

## Scope

The research platform is virtual. Physical robots, real sensors and neuromorphic
hardware are not current milestones. The practical target is an adaptive decision
core that can learn continuously inside games, simulated embodied worlds and
other virtual environments.

DOOMFLY remains the first reference implementation because it already provides a
reviewable MaleCNS v1.0 importer, full retained sparse graph, native fixed-step
runtime, checkpointing, a visual path, neuromodulation hooks and preserved
negative results. It is not intended to become the final general-purpose runtime.
This repository keeps its existing MaleCNS/ViZDoom scope; extraction to a general
virtual-world core happens only after a learning mechanism survives controlled
validation.

## Research question

Can a biologically constrained sparse recurrent network, using only local online
plasticity and neuromodulatory feedback, produce useful continual adaptation in a
changing virtual world without backpropagation through time or replaying a stored
training dataset?

The long-term engineering question is stronger: which parts of the full
connectome and learning dynamics are actually necessary, and can they be reduced
to a smaller reusable adaptive SNN core?

## Rules for evidence

1. A changed weight is not evidence of learning.
2. Longer Doom survival is not evidence of learning by itself.
3. Every learning claim needs a frozen control and a contingency-breaking control.
4. Sensory conditions must be distinguishable and activity-balanced before a
   conditioning result is interpreted.
5. Environment state may be used for scoring and reinforcement generation, but it
   must not secretly select actions.
6. When comparing learning rules, graph, sensory encoding, decoder, training
   exposure and evaluation protocol stay fixed unless the experiment explicitly
   studies one of those factors.
7. Negative results remain first-class outputs.

## Phase 0 — trustworthy baseline (current)

Goal: turn the existing v6 model into an interpretable associative-learning test.

Implemented on `research/snn-baseline`:

- pure decision/protocol tests;
- visual cue preflight;
- exact v5-protocol replication on v6;
- timing- and dose-matched strict differential conditioning;
- frozen, unpaired, no-US and memory-erasure controls;
- automatic local run report that routes failures to sensory, protocol/state or
  learning-rule work;
- one-command local WSL/Linux runner.

Exit gate:

- at least one cue pair passes sensory preflight;
- both CS+ counterbalances show positive paired selectivity;
- paired selectivity exceeds unpaired and no-US;
- frozen weights stay unchanged;
- memory erasure restores the original response;
- the result survives more than one eligible cue pair or other parameter-preserving
  replication before being treated as robust.

If the sensory preflight fails, plasticity is not tuned. If protocol/state checks
fail, the experiment infrastructure is fixed before changing the rule.

## Phase 1 — local learning-rule comparison

Goal: identify whether the useful mechanism is the current centered rule or a
more general local three-factor rule.

Compare on the same existing KC->MBON11 edges and the same strict protocol:

- fixed weights;
- current v6 centered anti-Hebbian rule;
- explicit eligibility + modulatory three-factor rule;
- reward-modulated STDP candidate if the spike timing available from the runtime
  is sufficient for a clean implementation.

Primary metrics:

- cue-specific effect size relative to unpaired/no-US;
- number and fraction of changed synapses;
- saturation fraction;
- retention after delayed dark intervals;
- erase/recovery behavior;
- simulated neural time and wall-clock cost.

Parameter search is deliberately small and declared in advance. A rule is not
selected because it happens to improve Doom survival on one run.

Exit gate: one rule repeatedly passes associative controls without relying on a
single cue identity or pathological saturation.

## Phase 2 — virtual continual-adaptation benchmark

This phase should live in a new general research repository rather than expanding
DOOMFLY beyond its stated scope.

Start with a deterministic, fully instrumented virtual world whose difficulty is
between conditioning and Doom. It should expose only defined sensory channels to
the SNN and keep privileged state for scoring/reinforcement only.

Recommended task ladder:

1. two-choice approach/avoidance;
2. resource acquisition with locations that change during a run;
3. navigation with changing hazards/reward locations;
4. pursuit/evasion with opponent-policy changes;
5. compact combat environment;
6. persistent NPC world.

The important benchmark is non-stationarity: the environment changes after the
agent has already adapted. Measure recovery without resetting weights.

Metrics:

- reward/task success before and after a distribution change;
- recovery time after the change;
- old-skill retention after learning the new condition;
- number of synaptic updates;
- active-neuron/active-edge fraction;
- wall time and memory;
- behavior variance across controlled seeds.

## Phase 3 — connectome attribution and distillation

Goal: determine whether MaleCNS topology contributes useful structure beyond the
learning rule itself.

Required comparisons:

- full MaleCNS topology + selected rule;
- degree/statistics-matched randomized topology + same rule;
- progressively ablated MaleCNS circuits + same rule;
- distilled subgraph + same rule.

Ablation/distillation order:

1. measure task-relevant activity and plastic-edge influence;
2. remove or silence candidate neuron classes/circuits one factor at a time;
3. identify motifs whose removal consistently destroys adaptation;
4. construct the smallest retained graph that preserves the effect;
5. compare the distilled graph against size/degree-matched synthetic graphs.

The desired result is not merely a smaller fly. It is a reusable sparse adaptive
architecture with evidence showing which biological structural priors matter.

## Phase 4 — general virtual adaptive intelligence core

Only after Phases 1-3 establish a useful mechanism should the reusable core be
extracted.

Proposed boundaries:

```text
Virtual environment
        |
Observation encoder
        |
Adaptive SNN core
        |
Action decoder
        |
Virtual environment
        ^
        |
reinforcement / neuromodulation
```

The core should separate:

```text
graph / topology
neuron dynamics
synapse dynamics
plasticity
neuromodulation
state/checkpoint
observation encoders
action decoders
experiment/audit records
```

Environment adapters then become replaceable modules: Doom, a small research
world, Godot NPC environments and later multi-agent worlds.

## Phase 5 — practical virtual-world applications

### Adaptive game NPC

Test whether identical initial NPCs develop different but stable behavior from
different experience histories. Candidate properties include risk preference,
route preference, target preference, retreat thresholds and combat habits.

The useful criterion is not human-like dialogue. It is persistent, measurable
behavioral adaptation that remains controllable by game design constraints.

### Simulated embodied agent

Use a richer virtual body and sensory model to study long-horizon navigation,
resource use, avoidance and recovery from changed dynamics. This remains entirely
simulated.

### Multi-agent virtual ecology

Only after single-agent continual adaptation is stable, evaluate multiple adaptive
agents with competition/cooperation and persistent individual memory. The focus is
emergent decision dynamics, not adding language models to hide weak SNN behavior.

## What the user needs to run

The intended workflow is local execution and verification only. Code, protocols,
analysis and iteration remain repository-managed.

For the current phase, from WSL2/Ubuntu on the research branch:

```sh
bash scripts/snn_research_local.sh all first-local
```

After completion (or an intentional preflight block), the most useful artifact to
return is:

```text
outputs/doom-learning/research-runs/first-local/research-report.json
```

If a stage crashes, preserve the run directory and provide the terminal traceback;
do not delete partial evidence before diagnosis.
