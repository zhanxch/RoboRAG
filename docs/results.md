# Pilot results

## Stack-blocks development success

The frozen `scene-phase-transition-v3` RoboRAG controller completed
`stack_blocks`, standard layout 0, with native **100/100 and success=true**.
It used 195 executed steps out of the native 550-step limit and 20 model calls.
The long-term bank contained three published training episodes and 58 valid
chunks. Each decision selected up to eight training chunks and four live chunks,
with compressed execution summaries retained separately.

This is one development success and establishes a working control pipeline.
It does not establish an advantage over fixed context. A matched fixed-context
run on layout 0 and a frozen-controller run on predeclared validation layout 1
are in progress. The [validation plan](../experiments/stack-v3-validation-plan.json)
was recorded before inspecting or executing layout 1. Native evidence is in
[the current pilot manifest](../results/pilot-current.json).

The revised deposit-coin development attempt was deliberately stopped after
110 observed steps, following repeated failed grasp attempts, to reallocate GPU
resources to the successful task's controls and validation. It has **no final
native score**, is recorded as an experimenter interruption, and is retained in
the evidence manifest. No online evaluation trajectory was added to the frozen
training bank.

## First deposit-coin development pair, 2026-10-16

The fixed-context control scored **20/100**, with no full success. The first
dual-memory selector scored **0/100**, also with no full success. Each condition
contains one completed rollout on the same development layout 0. These results
do **not** establish a retrieval benefit; the initial selector performed worse
on this pair.

Both conditions used `gpt-6-astra`, medium reasoning, the same account-based
transport, the same single demonstration source, six training chunks, four
live chunks, and RoboDojo's unchanged 300-step limit. Native results and hashes
are preserved in [the evidence manifest](../results/deposit-v1-development.json).
The score of 20 means the lifting threshold was reached, not that the coin was
deposited. Videos and complete logs are in the manifest's local artifact directories.

RGB review showed that the selected training example had the bank and coin on
opposite sides to the live scene. Absolute end-effector similarity repeatedly
retrieved the source's late insertion phase during live grasp attempts. Both
controllers knocked the coin onto the table and spent much of the episode
trying to recover it. A separate implementation review found that omitted live
intervals lost their textual execution summaries.

The subsequent development revision adds initial scene selection from a multi-demo
bank, an observation-based source-phase cursor, preserved compressed execution
summaries, and an explicit TRAIN→TEST calibration boundary. The stack success
above does not isolate the contribution of each change. Comparisons must use matching bank, model,
context quotas and code revision. Layouts used for tuning remain development
data even if a later rollout succeeds on them.

Limits: one development layout per condition; no uncertainty estimate or
generalization claim; local asset tree not newly certified against the complete
upstream asset revision; transport differs from the paper API; concurrent wall
times are not a matched latency measurement. Capture tests and incomplete
startup attempts are excluded from scored trials.
