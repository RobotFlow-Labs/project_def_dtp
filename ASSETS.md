# DEF-DTP — Asset Manifest

## Paper
- Title: DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction
- Correct arXiv: 2603.26462
- Incorrect scaffold arXiv to replace: 2503.15832
- Authors: Jiaxiang Li, Jun Yan, Daniel Watzenig, Huilin Yin
- PDF: `papers/2603.26462_DTP-Attack.pdf`
- Local stale PDF to ignore: `papers/2503.15832_DTP-Attack.pdf`

## Status: ALMOST

Paper and reference code are present, but benchmark datasets and pretrained predictor checkpoints are not staged in this repo yet.

## Reference Assets
| Asset | Source | Local Path | Status |
|---|---|---|---|
| Correct paper PDF | arXiv | `papers/2603.26462_DTP-Attack.pdf` | DONE |
| Reference repo | github.com/eclipse-bot/DTP-Attack | `repositories/DTP-Attack/` | DONE |
| Paper figures | Reference repo | `repositories/DTP-Attack/image/` | DONE |
| Wrong PDF artifact | stale scaffold | `papers/2503.15832_DTP-Attack.pdf` | STALE |

## Pretrained Weights
The paper evaluates five predictor settings: Grip++ on nuScenes and Apolloscape, Trajectron++ on nuScenes and Apolloscape, and Trajectron++ with maps on nuScenes. No direct checkpoint download URLs are published in the paper or repo README; the reference code expects them to exist at the following paths.

| Model | Size | Source | Path on Server | Status |
|---|---:|---|---|---|
| Grip++ / nuScenes | not stated | reproduce from Grip++ training recipe | `/mnt/forge-data/models/def_dtp/grip_nuscenes/original/best_model.pt` | MISSING |
| Grip++ / Apolloscape | not stated | reproduce from Grip++ training recipe | `/mnt/forge-data/models/def_dtp/grip_apolloscape/original/best_model.pt` | MISSING |
| Trajectron++ / nuScenes | not stated | reproduce from Trajectron++ training recipe | `/mnt/forge-data/models/def_dtp/trajectron_nuscenes/original/` | MISSING |
| Trajectron++ / Apolloscape | not stated | reproduce from Trajectron++ training recipe | `/mnt/forge-data/models/def_dtp/trajectron_apolloscape/original/` | MISSING |
| Trajectron++(map) / nuScenes | not stated | reproduce from Trajectron++ training recipe | `/mnt/forge-data/models/def_dtp/trajectron_map_nuscenes/original/` | MISSING |

## Datasets
The paper reports experiments on nuScenes and Apolloscape. The reference repo also contains NGSIM preprocessing support, but NGSIM is not part of the paper's main evaluation and should not be treated as a reproduction blocker.

| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| nuScenes prediction export | not stated in paper | train / val / test | [nuScenes](https://www.nuscenes.org/nuscenes) via Trajectron++ preprocessing | `/mnt/forge-data/datasets/def_dtp/nuscenes/prediction_{train,val,test}` | MISSING |
| Apolloscape trajectory prediction | not stated in paper | train / val / test | [Apolloscape trajectory dataset](http://apolloscape.auto/trajectory.html) | `/mnt/forge-data/datasets/def_dtp/apolloscape/prediction_{train,val,test}` | MISSING |
| nuScenes map assets | not stated | map metadata | nuScenes map bundle | `/mnt/forge-data/datasets/def_dtp/nuscenes/maps` | MISSING |
| Scenario sample manifest | 100 random samples per dataset | evaluation subset | generated locally from benchmark export | `artifacts/sample_manifests/{dataset}_100.txt` | MISSING |

## Hyperparameters (from paper)
This paper is an attack paper, not a training paper. The most important published hyperparameters are attack, dataset-window, and criteria thresholds.

| Param | Value | Paper Section |
|---|---|---|
| `LI` (nuScenes) | `4` | §V-A |
| `LO` (nuScenes) | `12` | §V-A |
| `LI` (Apolloscape) | `6` | §V-A |
| `LO` (Apolloscape) | `6` | §V-A |
| orthogonal step size `delta` | `1.0` initial | Algorithm 1 / §V-A |
| forward step size `epsilon` | `0.1` initial | Algorithm 1 / §V-A |
| orthogonal adjustment factor | `0.95` | Algorithm 1 / §V-A |
| forward adjustment factor | `0.9` | Algorithm 1 / §V-A |
| max iterations / query budget | `1000` | Algorithm 1 / §V-A |
| `theta_err_ADE` (nuScenes) | `7.5 m` | §V-A |
| `theta_err_ADE` (Apolloscape) | `3.5 m` | §V-A |
| `theta_err_FDE` (nuScenes) | `17.5 m` | §V-A |
| `theta_err_FDE` (Apolloscape) | `7.5 m` | §V-A |
| `theta_int_lateral` | `2.0 m` | §V-A |
| `theta_int_longitudinal` | `3.0 m` | §V-A |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---:|---:|
| Intention misclassification | ASR range | `41%` to `81%` | within paper range on matched setup |
| Prediction degradation | ADE increase | `1.9x` to `4.2x` | within `10%` of paper factor |
| Prediction degradation | FDE increase | `1.8x` to `3.9x` | within `10%` of paper factor |
| Stealthiness | perturbation MSE | `0.12 m` to `0.45 m` | within paper range |
| Directional attack | lateral deviation | about `2 m` | at or above threshold |
| Directional attack | longitudinal deviation | about `3 m` | at or above threshold |

## Hardware Requirements
- Paper hardware is not reported.
- Reference repo assumes CUDA throughout and hard-codes `.cuda()` in attack and adapter code paths.
- Reproduction target:
  - baseline parity: 1 CUDA GPU with enough memory for Trajectron++ inference and repeated query loops
  - ANIMA adaptation: optional MLX/CPU abstraction after CUDA-faithful baseline is validated

## Notes
- The project scaffold currently carries `BISHAMONTEN` naming and the wrong arXiv ID. PRD-01 must correct package, config, and docs to `DEF-DTP` and `2603.26462`.
- FQA exists in the upstream repo but is not part of the paper's reported main results. Keep it out of the first reproduction milestone.
