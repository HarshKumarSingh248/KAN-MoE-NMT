# Inference and evaluation

## Commands and checkpoint format

The evaluator takes a `best_model2.pt` state dict and the matching language configuration. For a checkpoint produced by the Hindi training script:

```bash
python evaluate.py --config configs/hindi.yaml \
  --checkpoint runs/hindi/best_model2.pt --split test --metric-protocol legacy
python evaluate.py --config configs/hindi.yaml \
  --checkpoint runs/hindi/best_model2.pt --split challenge --metric-protocol legacy
```

Valid splits are `dev`, `test`, and `challenge`. Paths and batch size can be overridden with `--data-dir`, `--nllb-dir`, `--output-dir`, and `--batch-size`. Results are saved as `runs/<language>/evaluation_<language>_<split>_<metric-protocol>.json`.

## Generation

Generation uses the configured NLLB target-language token, five-beam search, and the text-plus-bounding-box input. The default evaluation batch size is 16.

## Text normalization and metrics

Run `bash scripts/setup_indic_nlp.sh` after activating the environment. Hypotheses and references are normalized and tokenized with the pinned Indic NLP Library; missing dependencies cause an error.

The default `advisor` profile uses case-sensitive Moses `multi-bleu.perl` and RIBES 1.02.4 with `-c`, following the advisor-specified commands. Indic preprocessing uses the library API so input and output lines stay aligned.

The bundled Moses script matches `mosesdecoder` `RELEASE-2.1.1`. For the default profile, download [RIBES 1.02.4 from NTT](https://www.rd.ntt/cs/team_project/icl/lirg/ribes/) and place `RIBES.py` at `third_party/RIBES-1.02.4/RIBES.py`.

Use `--metric-protocol legacy` for the completed Hindi runs shown in the README. It uses Moses `-lc` and the bundled RIBES 1.03 without `-c`. Results record the selected profile; BLEU is rounded to two decimals and RIBES to four.

These local profiles are not claimed to replicate the WAT 2025 submission server, whose [findings paper](https://aclanthology.org/anthology-files/pdf/wat/2025.wat-1.10.pdf) reports SacreBLEU.

Generation uses the NLLB tokenizer; metric tokenization is applied only when scoring. References retain the TSV row order.
