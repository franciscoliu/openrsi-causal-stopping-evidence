# Causal stopping proposal evidence

Contributor-written diagnostic scripts and contributor-reported aggregate evidence for **Learned Causal Stopping for Safety Guardrails**, an OpenRSI task proposal by Zheyuan Liu. This is supplementary evidence, not an accepted OpenRSI task or a completed benchmark implementation.

## Scope

The 2026-10-02 pilot used 32 development responses, with 15 harmful and 17 benign responses under the declared annotations. No policy was trained and no test response received model inference. An all-split structural and tokenization audit preceded the development-only model check.

The pilot used full-sequence guard inference. Its measurements do not reproduce the proposal's canonical incremental evaluator, establish learned-policy improvement, demonstrate population safety, or estimate complete Work/Judge runtime. The reset diagnostic repeats no-cache guard inference; it does not validate resetting an incremental cache or recurrent policy state. FineHarm's LLM sentence labels and POS-derived word labels define an annotation proxy, not independently validated harmful information.

Only the three scripts and seven aggregate/hash reports listed here are included. No model assets, raw dataset rows, per-example test identifiers or traces, selected input-token arrays, token masks, credentials, cluster launch scripts, or conversation logs are included.

## Files and provenance

- `verification/audit_fineharm.py`: structural alignment and split-priority audit.
- `verification/prepare_pilot.py`: hardened post-run preparation source.
- `verification/run_guard_pilot.py`: original diagnostic full-sequence replay and limited incremental comparisons; this is not the final task evaluator.
- `verification/crc_results_20261002/asset_manifest.json`: acquired asset revisions, sizes and SHA-256 values.
- `preparation_report.json`, `guard_pilot.json`, and `source_sha256.txt` in that reports directory: original-run aggregates and source hashes.
- `preparation_report_checked.json`, `boundary_reaudit.json`, and `source_sha256_checked.txt`: later structural re-audit without new guard inference or sample replacement.

The included preparation source has SHA-256 `5e6c44d05f70c028689e05d2b0779f2a17f3b5e444497b11897c87ded45051b4`. It is not the original preparation source, whose recorded hash is `5c2fa867a69d95a3187a95f2f8c1209d29d5b2c4c156d474c6677f750ae0ddbf` and whose bytes are not included. The included structural-audit and guard-pilot scripts match both historical source manifests. Those historical manifests also name files omitted from this small evidence set.

The boundary re-audit reports unchanged private records, eligibility, causal IDs and tolerances. Different selection-file hashes include changed metadata. This public bundle does not expose the private records or independently prove those equality claims; it preserves the contributor's reports and their limitations.

Reproduction requires authorized acquisition and private preparation of the declared assets. This bundle is not a standalone data-containing environment. Do not publish the scripts' generated selected inputs: token IDs can reconstruct gated content.

## External sources and terms

- [Qwen3Guard source](https://github.com/QwenLM/Qwen3Guard/tree/6a52eca94b3d2aedb8aebd36baa353828d4166f1), including the upstream streaming evaluator and unsafe-pair rule.
- [Qwen/Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B/tree/74e1479150e9029d6778993f00491108323bb6f8), model/tokenizer/custom-code revision `74e1479150e9029d6778993f00491108323bb6f8`. Its inspected license and required custom-code headers are Apache-2.0.
- [liyang-ict/FineHarm](https://huggingface.co/datasets/liyang-ict/FineHarm/tree/e20a087e2ec0c5550f2a53a508503bb9a6741752), dataset revision `e20a087e2ec0c5550f2a53a508503bb9a6741752`.
- [ICTMCG/SCM preprocessing reference](https://github.com/ICTMCG/SCM/blob/b0e4ab48011a0441767fbbd3537266498e874eeb/utils/data_process.py), supporting the declared annotation schema.
- FineHarm source datasets: [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) and [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak).

FineHarm declares MIT, while its identified WildGuardMix/WildJailbreak source chain also carries ODC-BY attribution obligations and Ai2 responsible-use terms. Reproduction must use authorized access and retain applicable notices; contributor access does not establish another runner's authorization. These source references do not grant permission to redistribute gated examples or underlying content. No upstream code, weights or data files are copied into this evidence bundle.
