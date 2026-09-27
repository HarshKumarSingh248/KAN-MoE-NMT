# Dataset and preprocessing

Download the language-specific Visual Genome translation data from the [WAT 2025 English–Indic Multimodal Translation task page](https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation). The data are not redistributed here. After extraction, run:

```bash
bash scripts/prepare_data.sh LANGUAGE /path/to/extracted/archive
```

The script copies the `train`, `dev`, `test`, and `challenge-test-set` TSVs into `data/<language>/`. Expected filenames are `<language>-visual-genome-<suffix>.txt`. Each UTF-8 row has seven tab-separated fields: `image_id`, `x`, `y`, `w`, `h`, English text, and target text, with no header. Check split counts against the downloaded archive; the Hindi record contains 28,930/998/1,595/1,400 rows in that order.

The active data loader converts the four box values to numbers, substitutes zero for invalid values, divides `x` and `w` by 640 and `y` and `h` by 480, then clips each value to `[0,1]`. It uses source/target text and box coordinates, not image pixels. English is tokenized as `eng_Latn`; the target language comes from the language config. Metric-specific text preprocessing is in [Evaluation](EVALUATION.md).
