# Reproducibility

Use the language configs and the procedures in [Training](TRAINING.md), [Dataset](DATASET.md), and [Evaluation](EVALUATION.md) to run the released implementation. The configured backbone is `facebook/nllb-200-1.3B`; the model uses bounding-box coordinates but does not load image pixels.

The Hindi example in the README is the mean of three completed five-epoch full-system runs, scored with the `legacy` metric profile. It is separate from the submitted manuscript's results. The manuscript-matching Hindi checkpoint and Bengali run artifacts are not included here, so this repository alone cannot independently reproduce every reported manuscript score. Malayalam is an additional configuration, not a submitted-paper result.

For exact comparisons, record the dataset archive and checksums, backbone revision, Indic NLP resources revision, environment, seed, checkpoint, and metric profile. The original backbone and Indic NLP resource revisions were not recorded in the available run artifacts. A saved project checkpoint contains model weights, not optimizer or scheduler state, so it does not support an exact training resume.
