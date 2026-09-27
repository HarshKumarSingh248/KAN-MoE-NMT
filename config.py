import os
from dataclasses import dataclass


@dataclass
class CFG:
    lang: str = "hindi"
    src_lang_nllb: str = "eng_Latn"
    tgt_lang_nllb: str = "hin_Deva"
    lang_code: str = "hi"

    img_ref_w: int = 640
    img_ref_h: int = 480

    data_dir: str = ""
    tsv_prefix: str = "hindi-visual-genome"

    nllb13b_local: str = ""
    work_dir: str = ""

    ribes_script: str = ""
    indic_nlp_dir: str = ""
    metric_protocol: str = "advisor"  # "advisor" or "legacy" (historical local scores)

    d_model: int = 1024
    kan_hidden: int = 2048
    kan_basis: int = 16
    n_experts: int = 4
    kan_dropout: float = 0.1
    moe_aux_wt: float = 0.05
    fourier_freqs: int = 16

    expert_type: str = "rbf"       # "rbf" | "mlp" (param-matched) | "spline"
    use_region_gate: bool = True   # False → w/o RegionGate ablation
    use_aux: bool = True           # False → w/o Aux Loss ablation (α=0)
    use_residual: bool = True      # False → w/o Residual-in-KAN-MoE ablation
    raw_coords: bool = False       # True  → w/o Fourier ablation (raw bbox coords)

    epochs: int = 30
    batch_size: int = 8
    grad_accum: int = 4            # effective batch 32
    lr_kan: float = 3e-4
    lr_decoder: float = 1e-5
    lr_encoder: float = 1e-5
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    patience: int = 5
    eval_every: int = 1
    seed: int = 42

    max_src_len: int = 128
    max_tgt_len: int = 128
    beam_size: int = 5
    length_penalty: float = 1.0
    no_repeat_ngram: int = 3
    repetition_penalty: float = 1.15

    num_workers: int = 4
    pin_memory: bool = True


def get_cfg(lang: str, data_dir: str, nllb13b_local: str,
            work_dir: str, ribes_script: str, indic_nlp_dir: str) -> CFG:
    if lang == "hindi":
        cfg = CFG(
            lang="hindi",
            tgt_lang_nllb="hin_Deva",
            lang_code="hi",
            tsv_prefix="hindi-visual-genome",
        )
    elif lang == "bengali":
        cfg = CFG(
            lang="bengali",
            tgt_lang_nllb="ben_Beng",
            lang_code="bn",
            tsv_prefix="bengali-visual-genome",
        )
    elif lang == "malayalam":
        cfg = CFG(
            lang="malayalam",
            tgt_lang_nllb="mal_Mlym",
            lang_code="ml",
            tsv_prefix="malayalam-visual-genome",
            patience=10,
        )
    else:
        raise ValueError(f"Unknown language: {lang}")

    cfg.data_dir       = data_dir
    cfg.nllb13b_local  = nllb13b_local
    cfg.work_dir       = work_dir
    cfg.ribes_script   = ribes_script
    cfg.indic_nlp_dir  = indic_nlp_dir
    return cfg
