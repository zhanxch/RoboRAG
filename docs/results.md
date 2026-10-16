# Pilot results

## Stack-blocks pilot

The frozen `scene-phase-transition-v3` RoboRAG controller completed
`stack_blocks`, standard layout 0, with native **100/100 and success=true**.
It used 195 executed steps out of the native 550-step limit and 20 model calls.
The long-term bank contained three published training episodes and 58 valid
chunks. Each decision selected up to eight training chunks and four live chunks,
with compressed execution summaries retained separately.

The matched fixed-context control also succeeded on layout 0, scoring 100/100
in 240 steps and 24 model calls. Observed token totals were 1,289,773 for RoboRAG
and 1,506,661 for fixed context. This one development pair shows fewer calls for
RoboRAG, not a statistically established efficiency or success-rate advantage.

With the controller and bank frozen, RoboRAG also succeeded on predeclared
validation layout 1: **100/100, success=true, 302 steps and 31 calls** (2,048,690
observed tokens). A placement became unstable around steps 110–130; the policy
regrasped the wood block and completed the stack. At step 160, short-term retrieval
included the earlier placement interval 110–120, outside a latest-four-chunk
window, while retaining the latest interval 150–160. This verifies retrieval of
older relevant experience but does not by itself establish a causal contribution
to recovery. The fixed-context validation control also scored **100/100 with
success=true**, in **237 steps and 24 calls** (1,509,991 observed tokens).

Both methods therefore succeeded in both selected layouts. RoboRAG used fewer
steps/calls in development (195/20 versus 240/24), but more in validation
(302/31 versus 237/24). These observations show neither a success-rate gain nor
a consistent efficiency advantage for retrieval. The result is a working
dual-memory control pilot, not evidence that this selector outperforms fixed
demonstrations. The four trials share identical frozen corpus, retrieval-controller
source and harness hashes; their per-method context selection differs as intended.

Thus RoboRAG has two completed successful pilot rollouts: one development layout
and one layout not used for this project's tuning. They are not two independent
held-out trials and do not constitute a benchmark-wide result.
The [validation plan](../experiments/stack-v3-validation-plan.json)
was recorded before inspecting or executing layout 1. Native evidence is in
[the current pilot manifest](../results/pilot-current.json).

The revised deposit-coin development attempt was deliberately stopped after
110 observed steps, following repeated failed grasp attempts, to reallocate GPU
resources to the successful task's controls and validation. It has **no final
native score**, is recorded as an experimenter interruption, and is retained in
the evidence manifest. No online evaluation trajectory was added to the frozen
training bank.

All pilot simulator and model-server processes have exited. The current repository
passes 23 software regression tests, covering memory boundaries, evidence
classification and cancellation cleanup. These tests are separate from the native
robot-task outcomes above.

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

Limits: one rollout per condition/layout; no uncertainty estimate or broad
generalization claim; the [28-file asset subset audit](../results/asset-scope-audit.json)
passed, but the complete asset tree was not newly certified; transport differs from the paper API; concurrent wall
times are not a matched latency measurement. Capture tests and incomplete
startup attempts are excluded from scored trials.
