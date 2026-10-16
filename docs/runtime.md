# RoboDojo pilot runtime

## Dependency boundary

The pilot reuses RoboICL's model-facing observation/action format, IK checks,
camera calibration, and RoboDojo execution adapter. RoboRAG owns the offline
corpus, online retrieval, separate episode memory selection, and experiment
launcher. The motivation is written around ICL and retrieval; implementation
provenance remains explicit.

- RoboICL: `07d6e1c5387710e2de9240724b0015bbf0c95c6e`, [MIT source](https://github.com/Mosi-AI/RoboICL).
- RoboDojo: `00ef04cd245bfa76e6f1e0bd8383c82f7964f942`.
- XPolicyLab: `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- IsaacLab: `afca7b09d60d8beb9c1cb28b43066499940b969b`.
- cuRobo: `895c6517243f8cb091c73c018c8167192d39599a`.

The latter four are pinned by the upstream lock file and checked before every
launch. Dependencies are kept under ignored `vendor/`; their original licenses
remain in the dependency checkouts. No benchmark scoring or task source is edited.

The Codex CLI transport was copied from the user's existing local
`infiniDojo/reproduction/policy_compare/{codex_inference,roboicl_codex_transport}.py`
adaptation, using a package-relative import. The current bridge also unwinds
in-flight inference on SIGTERM so canceled runs do not orphan CLI processes;
the four frozen stack trials were launched before this lifecycle-only fix.
It uses the
existing signed-in CLI without exporting credentials. It serializes role history,
attaches images, disables agent tools and validates the returned action JSON.
It is **not** the original paper's API condition. The first fixed/dual runs use
the original local transport files; subsequent runs use the copied equivalent.

## Local setup

1. Prepare Isaac Sim 5.1 and the policy interpreter using the upstream installation instructions.
2. Clone RoboICL into `vendor/RoboICL`, check out the commit above, and initialize its pinned submodules. On this host, identical clean dependency checkouts are reused via links to save disk space.
3. Copy `local.runtime.example.json` to ignored `local.runtime.json` and set interpreter, library and asset paths.
4. Set `data/Assets` to the selected asset installation. This host reuses its existing corrected asset tree; run manifests record the actual asset path and layout hash. It has not been newly certified as a complete upstream asset revision.
5. Use upstream `setup/fetch_reference.py deposit_coin --data-root /absolute/path/to/RoboRAG/data` with the policy interpreter. The published source HDF5 is checked by SHA256.
6. Run `scripts/build_corpus.py --task deposit_coin` with the policy interpreter to build the dense corpus.

The visual baseline uses torchvision's frozen `ResNet18_Weights.IMAGENET1K_V1`
(checkpoint `resnet18-f37072fd.pth`), standard preprocessing, three separately
normalized view vectors, plus measured robot end-effector and gripper state.
It is a first retrieval baseline, with no claim of optimal visual representation.
The encoder runs on CPU, leaving GPU memory to simulation.

The v2 bank supports multiple pinned training episodes. Use
`scripts/fetch_demonstrations.py --task stack_blocks --episodes 0,1,2`, then
`scripts/build_visual_bank.py --manifest data/runtime-data/stack_blocks/train/sources-0-1-2.json --output data/corpora/stack_blocks/v2.json`.
Each source is checked against the fixed-revision Hugging Face LFS SHA256, its
25 Hz frequency, camera arrays and action bounds. Published trajectories are
not assigned a replay-verified success label without such evidence.

V2 chooses one coherent source episode from ordered 2×2 head-image ResNet
features at live step zero; that choice is frozen within the episode. Both
`fixed` and retrieval methods share this source selection. Dynamic retrieval
then compares the three-view features and robot state within that source,
reducing the state weight from 0.5 to 0.2 because absolute arm positions can
confuse different layouts and manipulation roles. Training records are keyed
by source hash and interval; equal frame numbers from separate episodes cannot
overwrite each other. This spatial descriptor is a baseline, not a validated
object/role representation.

The next controller revision (`scene-phase-transition-v3`) also aligns a source
phase cursor from the latest RGB/state match, permitting two horizons of
backtracking and four horizons of forward matching per new observation.
Retrieval applies a soft distance-from-phase prior while retaining initial and
terminal anchors. This prevents a robot's initial rest pose from preferentially
retrieving return-to-home actions at the end of the source trajectory. The cursor
uses observed similarity, not the evaluation step count or hidden task progress.
Fixed selection remains uniform. This is an additional development hypothesis
that needs live validation; both code revision and selector version are logged.

Observed policy environment: Python 3.11, torch 2.7.0+cu128, torchvision 0.22.0+cu128,
numpy 1.26.0, h5py 3.16.0, Pillow 12.3.0, scipy 1.15.3,
jsonschema 4.26.0 and websockets 15.0.1. Simulator and policy interpreters are
separate. Preserve the simulator's own dependency constraints.

## Run

From the repository root, using the configured policy interpreter:

```bash
python scripts/run_episode.py --mode zero --capture-only --output runs/camera-check
python scripts/run_episode.py --mode fixed --output runs/deposit-fixed-l0
python scripts/run_episode.py --mode dual --output runs/deposit-dual-l0
```

Output directories must be new. `--layout`, `--variant`, `--seed`, `--effort`,
`--long-k` and `--short-k` are recorded per run. Current defaults use six training
chunks and four live chunks (including the latest), with current/initial observations
also retained. The upstream serializer checks the actual image and text caps.
`fixed` uses uniform long-term chunks and recent live chunks; `dual` retrieves both;
`long_only` retrieves long-term chunks while retaining recent live chunks.

Each `dual` long-term selection includes the first and terminal training chunks
as procedural anchors and fills remaining slots by state/visual distance with
temporal diversity. The short-term index contains only this episode's committed
interactions. Initial and latest observations, action receipts and rejection tails
are retained; selected earlier interactions are presented chronologically.
From v2 onward, omitted live intervals retain their recorded execution/intent
summaries as text. They are explicitly marked as unverified intent rather than
success labels. The initial two deposit runs predate this correction; their
source hashes remain recorded, so their results must not be silently pooled
with later versions.

`--allow-shared-gpu` permits a second pilot only with at least 11,000 MiB free GPU
memory at launch. Overlapping process IDs are logged. Concurrent wall times must
not be interpreted as a matched latency comparison.

## Evidence

`run.json` records commands and configuration; `exit.json` records process status.
Native `_result.json` distinguishes progress score and full success. Model API
errors, setup failures and interrupted episodes are not successful evaluations.
`policy/*/retrieval.jsonl` records selected training intervals, short-term history
and corpus hash. `decisions/call-*` preserves supplied images, actions and usage.
All large artifacts stay under ignored `runs/` and `data/`.

The legacy `RoboICL` segment in native output paths comes from the reused
simulator adapter's policy label. The actual RoboRAG selector, model configuration
and source hashes are recorded by the enclosing run manifest and retrieval log.

A [partial asset audit](../results/asset-scope-audit.json) matched all 28 checked
robot URDF/config/mesh, block variant and layout files to the upstream frozen
asset revision. No assets were changed during this read-only audit. This is not
a complete transitive audit of all room, material, sensor and generated files.

Use `--role validation` only after freezing the policy and bank and selecting
layouts not used for development. The default is `development`. Export compact
native evidence with `scripts/export_evidence.py runs/<name> ... --output results/<report>.json`.
This preserves native result JSON, hashes, source/configuration identity and
video paths. Interrupted policy scores and missing completion evidence remain
excluded from valid scores; a process exit code of zero is insufficient.
