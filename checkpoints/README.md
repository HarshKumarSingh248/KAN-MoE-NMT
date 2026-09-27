# Checkpoints

No trained checkpoint is distributed in this repository. NLLB backbone files
and fine-tuned checkpoints are large and subject to their respective model
licenses. After training, the reference script writes `best_model2.pt` under
the configured run directory.

To evaluate an externally released checkpoint, place it at the path documented
by its provider or pass that path with `--checkpoint`. A project checkpoint is
a PyTorch state dictionary for `KANMoEModel`; it is not a Hugging Face model
directory.
