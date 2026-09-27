# Training

Prepare the dataset and dependencies as described in the [README](../README.md), then run a language script:

```bash
bash scripts/train_hindi.sh
bash scripts/train_bengali.sh
bash scripts/train_malayalam.sh
```

The scripts use `configs/<language>.yaml`. Pass CLI options to change paths or runtime settings; use a separate output directory for each run. For a Hindi run scored like the three-run README example:

```bash
bash scripts/train_hindi.sh --metric-protocol legacy --output-dir runs/hindi_example
```

The default `advisor` metric profile and the historical `legacy` profile differ; see [Evaluation](EVALUATION.md). The selected checkpoint can change with the scoring profile because dev BLEU determines selection.

The release configuration uses full fine-tuning, AdamW, BF16 autocast, per-device batch size 8, gradient accumulation 4, cosine scheduling with 2,000 warmup steps, and dev-BLEU checkpoint selection. The language-specific epoch budgets and patience values are in the YAML files. The selected `best_model2.pt` contains model weights only; final test and challenge metrics are computed after loading it.
