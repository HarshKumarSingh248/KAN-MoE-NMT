# Geometry of Expert Choice in Neural Machine Translation

Research code for the manuscript “Geometry of Expert Choice in Neural Machine Translation,” submitted to Expert Systems with Applications. This release contains model, training, and evaluation code, language configurations, and instructions for obtaining data and the backbone. It does not redistribute datasets, NLLB weights, trained checkpoints, or historical run logs.

For result scope and reproducibility notes, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Overview

The system combines an NLLB-200-1.3B encoder-decoder with spatial conditioning and a mixture of experts in the encoder representation path. A RegionGate maps each record’s normalized bounding-box coordinates through Fourier features to an additive encoder bias. A softmax router mixes four radial-basis-function (RBF) KAN experts; the fusion block includes a residual connection and load-balancing auxiliary loss. The decoder generates the target translation.

The active path consumes English text and bounding-box coordinates, not image pixels. Although the source data come from a visual-genome translation task, this implementation should not be described as an image-encoder model. Configurations are provided for English→Hindi, English→Bengali, and English→Malayalam; a configuration does not imply a corresponding paper result or locally available checkpoint.

## Repository layout

    config.py, model.py, train.py, evaluate.py   model, configuration, training, evaluation
    data_utils.py, RIBES.py                      data loading and metric implementation
    configs/                                     per-language YAML settings
    scripts/                                     setup, data, training, evaluation helpers
    data/                                        destination; data are not redistributed
    models/                                      destination for downloaded NLLB weights
    checkpoints/                                 destination for trained checkpoints
    third_party/multi-bleu.perl                  Moses BLEU script used by evaluator
    docs/                                        dataset, training, evaluation, reproducibility

## Requirements

The pinned environment targets Linux, Python 3.10, PyTorch 2.0.0 with CUDA 11.7, and Transformers 4.36.2. Full fine-tuning of the 1.3B backbone is memory-intensive; the completed Hindi runs used an NVIDIA A100 80 GB. Allow sufficient disk for NLLB, dataset archives, and optimizer state.

## Installation

```bash
git clone https://github.com/HarshKumarSingh248/KAN-MoE-NMT.git
cd KAN-MoE-NMT
bash scripts/setup_environment.sh
conda activate kan-moe-nmt
```

Install the Indic NLP code and resources using the same layout as the working project. The helper pins the library to the recorded source commit, clones the separate resources repository, and installs the package into the active environment:

```bash
bash scripts/setup_indic_nlp.sh
```

Training and evaluation load the library and resources from `third_party/indic_nlp_library`.

For the default evaluation setting, obtain **RIBES.py 1.02.4** from the [NTT download page](https://www.rd.ntt/cs/team_project/icl/lirg/ribes/) and place it at `third_party/RIBES-1.02.4/RIBES.py`.

Acquire NLLB-200-1.3B from Meta’s official distribution or its Hugging Face model page, subject to its license and access terms. Save files under models/nllb-200-1.3B, or pass another location with --nllb-dir. The identifier used was facebook/nllb-200-1.3B; the original immutable revision was not recorded.

## Dataset

The WAT English–Indic Multimodal Translation task maintainers distribute language-specific Visual Genome TSVs. Start from the [official WAT 2025 task page](https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation) and follow its Hindi, Bengali, or Malayalam LINDAT links. Follow dataset access terms and citation requirements. The helper does not guess direct archive URLs.

```bash
bash scripts/download_data.sh hindi
# After downloading and extracting the official archive:
bash scripts/prepare_data.sh hindi /path/to/extracted/archive
```

Repeat for Bengali and Malayalam. Expected files under data/<language>/:

```text
<language>-visual-genome-train.txt
<language>-visual-genome-dev.txt
<language>-visual-genome-test.txt
<language>-visual-genome-challenge-test-set.txt
```

Each row has seven tab-separated fields: image id, x, y, width, height, English text, target text. Coordinates are divided by 640 (x/width) and 480 (y/height), then clipped to [0,1]. Invalid numeric coordinates become zero. No image files are loaded by the active path; no region filtering or enclosing-box construction is performed.

Paper Hindi split counts are 28,930 train, 998 dev, 1,595 test, and 1,400 challenge. The available Malayalam dataset record reports the same counts. Bengali data files were not available to verify their split counts. Verify counts in the archive you download.

## Model and configuration

The implementation is in model.py. Shared settings include d_model=1024, KAN hidden size 2048, 16 RBF bases, four experts, dropout 0.1, auxiliary-loss weight 0.05, and 16 Fourier frequencies. Target tokens are hin_Deva, ben_Beng, and mal_Mlym; source token is eng_Latn. YAML files expose model, training, and generation settings.

## Training

Prepare data, acquire NLLB, and install the evaluation dependencies. For the Hindi example results below, use:

```bash
bash scripts/train_hindi.sh --metric-protocol legacy
```

For multiple runs, use separate output directories and record each seed and configuration. Training scripts for Bengali and Malayalam are also provided in `scripts/`.

Each script reads configs/<language>.yaml. Override paths or settings without editing Python:

```bash
bash scripts/train_hindi.sh --data-dir /data/hindi --nllb-dir /models/nllb --output-dir runs/hindi_custom --metric-protocol legacy
```

Training selects the checkpoint with highest dev BLEU and writes best_model2.pt in the configured output directory. It computes test and challenge metrics after loading that checkpoint and writes final_results.json. Epoch budgets and patience differ by language; see docs/TRAINING.md. Training duration depends on hardware and is not estimated here.

## Evaluation

Provide a compatible model state-dict checkpoint. The training command above writes one to `runs/hindi/best_model2.pt` by default. For the Hindi example below:

```bash
bash scripts/evaluate_hindi.sh runs/hindi/best_model2.pt test --metric-protocol legacy
bash scripts/evaluate_hindi.sh runs/hindi/best_model2.pt challenge --metric-protocol legacy
```

The same evaluation wrappers are available for Bengali and Malayalam. See [docs/EVALUATION.md](docs/EVALUATION.md) for metric commands and options.

## Example Hindi results

Full-system mean over three completed five-epoch runs, evaluated with the `legacy` setting above:

| Split | BLEU | RIBES |
|---|---:|---:|
| Test | 45.24 | 0.8276 |
| Challenge | 55.82 | 0.8710 |

These are experimental run means, not values hard-coded in the evaluator. Their relationship to the submitted results is explained in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

If you find this work or code useful in your research, please cite the associated paper:

```bibtex
@article{singh2026geometry,
  title   = {Geometry of Expert Choice in Neural Machine Translation},
  author  = {Singh, Harsh Kumar and Gain, Baban and Kumar, Deepak and Ekbal, Asif},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Manuscript; update publication status and final metadata when available}
}
```

No DOI, volume, pages, article number, acceptance, or publication date is asserted. Update CITATION.cff and this entry when verified metadata becomes available.

## Availability and licensing

Datasets, NLLB weights, and trained checkpoints are not distributed here. Consult the repository license and the respective third-party terms before reuse.
