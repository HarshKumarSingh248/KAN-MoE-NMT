"""Train the NLLB-based KAN-MoE translation model."""

# Configure the runtime library path before importing native extensions.
import os, sys, warnings
_LIBSTDCXX_CANDIDATES = [
    os.environ.get("KAN_MOE_LIBSTDCXX_DIR", ""),
]
if not os.environ.get("_GLIBCXX_FIXED"):
    for _lib in _LIBSTDCXX_CANDIDATES:
        if os.path.isdir(_lib):
            os.environ["LD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["_GLIBCXX_FIXED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)

warnings.filterwarnings("ignore", message=".*tie.*word.*embeddings.*")
warnings.filterwarnings("ignore", message=".*tied weights.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import argparse
import json
import logging
import random
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import transformers
transformers.logging.set_verbosity_error()

from config import get_cfg, CFG
from data_utils import (
    load_split, HVGDataset, make_collate_fn,
    compute_bleu, compute_ribes, init_indic_nlp, validate_metric_setup,
)
from model import KANMoEModel


# ── Local repository paths ──────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_HERE, "data")
_NLLB13B_DIR = os.path.join(_HERE, "nllb13b_local")
_RIBES       = os.path.join(_HERE, "RIBES.py")
_RIBES_ADVISOR = os.path.join(_HERE, "third_party", "RIBES-1.02.4", "RIBES.py")
_INDIC_NLP   = os.path.join(_HERE, "third_party", "indic_nlp_library")

_WORK_DIR_HINDI   = os.path.join(_HERE, "runs", "hindi")
_WORK_DIR_BENGALI = os.path.join(_HERE, "runs", "bengali")

_BENGALI_DATA_DIR = os.path.join(_HERE, "data")


# ── Distributed helpers ────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── DataLoader ────────────────────────────────────────────────────────────────

def make_nllb_loader(df, tokenizer, cfg: CFG, split: str, distributed: bool = False):
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer,
        src_lang=cfg.src_lang_nllb,
        tgt_lang=cfg.tgt_lang_nllb,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
    )
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=(split == "train"),
    )


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_all(model: KANMoEModel, tokenizer, df, cfg: CFG, device):
    model.eval()
    dataset = HVGDataset(df)
    collate = make_collate_fn(
        tokenizer,
        src_lang=cfg.src_lang_nllb,
        tgt_lang=cfg.tgt_lang_nllb,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * 2,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    hyps, refs = [], []
    for batch in loader:
        preds = model.generate(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["bbox"].to(device),
            tokenizer,
        )
        hyps.extend(preds)
        refs.extend(batch["tgt_texts"])
    return hyps, refs


# ── Main training loop ────────────────────────────────────────────────────────

def run(cfg: CFG, resume_from: int = 0, resume_ckpt: str = None, best_bleu_override: float = -1.0):
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    if is_main(rank):
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(cfg.work_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w"),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    logger = logging.getLogger(__name__)

    if is_main(rank):
        logger.info("=" * 65)
        logger.info("KAN-MoE translation model training")
        logger.info(f"  Language  : {cfg.lang}")
        logger.info(f"  Work dir  : {cfg.work_dir}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Device    : {device}")
        logger.info(f"  Backbone  : NLLB-200-1.3B  (full finetune, no freezing)")
        logger.info(f"  KAN-MoE   : {cfg.n_experts} experts, hidden={cfg.kan_hidden}, basis={cfg.kan_basis}")
        logger.info(f"  Expert    : {getattr(cfg,'expert_type','rbf')}  "
                    f"[region_gate={getattr(cfg,'use_region_gate',True)} "
                    f"aux={getattr(cfg,'use_aux',True)} residual={getattr(cfg,'use_residual',True)} "
                    f"raw_coords={getattr(cfg,'raw_coords',False)}]")
        logger.info(f"  Seed      : {cfg.seed} | warmup={cfg.warmup_steps} patience={cfg.patience} "
                    f"max_len={cfg.max_src_len}")
        logger.info(f"  RegionGate: additive Fourier bias (before KAN-MoE)")
        logger.info(f"  LR        : kan={cfg.lr_kan}  decoder={cfg.lr_decoder}  encoder={cfg.lr_encoder}")
        logger.info(f"  Smoothing : 0.1 | beam={cfg.beam_size} | no_repeat={cfg.no_repeat_ngram}")
        logger.info("=" * 65)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.nllb13b_local)
    tokenizer.src_lang = cfg.src_lang_nllb
    tokenizer.tgt_lang = cfg.tgt_lang_nllb

    # ── Data ──────────────────────────────────────────────────────────────
    train_df = load_split(cfg, "train")
    dev_df   = load_split(cfg, "dev")

    train_loader = make_nllb_loader(train_df, tokenizer, cfg, "train",
                                    distributed=(world_size > 1))
    dev_loader   = make_nllb_loader(dev_df, tokenizer, cfg, "dev", distributed=False)

    if is_main(rank):
        logger.info(f"Train batches: {len(train_loader)}  Dev batches: {len(dev_loader)}")

    # ── Model (full finetune — no freezing) ───────────────────────────────
    model = KANMoEModel(cfg).to(device)

    if resume_ckpt and os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))
        if is_main(rank):
            logger.info(f"Loaded checkpoint: {resume_ckpt}")

    if is_main(rank):
        counts = model.param_counts()
        state_dict_count = sum(t.numel() for t in model.state_dict().values())
        logger.info(
            f"Params — unique: {counts['total']:,}  trainable: {counts['trainable']:,} | "
            f"state-dict entries (incl. tied embeddings): {state_dict_count:,}"
        )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw: KANMoEModel = model.module if hasattr(model, "module") else model

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        raw.get_param_groups(), weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6,
    )
    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    best_bleu        = best_bleu_override if best_bleu_override > 0 else -1.0
    patience_counter = 0
    history          = []
    ckpt_path        = os.path.join(cfg.work_dir, "best_model2.pt")

    if is_main(rank) and resume_from:
        logger.info(f"Resuming from epoch {resume_from + 1}, best_bleu={best_bleu:.2f}")

    # ── Training loop ─────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(resume_from + 1, cfg.epochs + 1),
        desc="Overall", disable=not is_main(rank),
        ncols=100, unit="ep", file=sys.stdout,
    )

    for epoch in epoch_bar:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        aux_accum  = 0.0
        optimizer.zero_grad()
        n_batches  = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{cfg.epochs}",
            disable=not is_main(rank),
            ncols=100, file=sys.stdout, dynamic_ncols=False,
        )

        for step, batch in enumerate(pbar, 1):
            inp  = batch["input_ids"].to(device, non_blocking=True)
            msk  = batch["attention_mask"].to(device, non_blocking=True)
            lbl  = batch["labels"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, aux_loss = model(
                    input_ids=inp, attention_mask=msk,
                    labels=lbl, bbox=bbox, label_smoothing=0.1,
                )

            scaler.scale(loss / cfg.grad_accum).backward()
            epoch_loss += loss.item()
            aux_accum  += aux_loss.item()

            if step % cfg.grad_accum == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if is_main(rank):
                pbar.set_postfix(
                    loss=f"{epoch_loss/step:.4f}",
                    aux=f"{aux_accum/step:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        pbar.close()
        avg_loss = epoch_loss / n_batches
        avg_aux  = aux_accum  / n_batches

        # ── Evaluation (rank 0 only) ───────────────────────────────────────
        if epoch % cfg.eval_every == 0 and is_main(rank):
            hyps, refs = generate_all(raw, tokenizer, dev_df, cfg, device)
            bleu  = compute_bleu(hyps, refs, cfg)
            ribes = compute_ribes(hyps, refs, cfg)

            logger.info(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"loss={avg_loss:.4f}  aux={avg_aux:.4f} | "
                f"Dev BLEU={bleu:.2f}  RIBES={ribes:.4f}"
            )
            history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu, "ribes": ribes})

            if bleu > best_bleu:
                best_bleu        = bleu
                patience_counter = 0
                torch.save(raw.state_dict(), ckpt_path)
                logger.info(f"  → Saved best_model2.pt  (BLEU={bleu:.2f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{cfg.patience})")
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping triggered.")

            epoch_bar.set_postfix(
                BLEU=f"{bleu:.2f}", best=f"{best_bleu:.2f}",
                patience=f"{patience_counter}/{cfg.patience}",
            )

        # Early-stop: sync across ranks when distributed, otherwise break directly.
        # Synchronize the early-stop decision across distributed workers.
        if world_size > 1:
            stop_flag = torch.tensor(
                int(patience_counter >= cfg.patience), device=device
            )
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                break
        elif patience_counter >= cfg.patience:
            break

    epoch_bar.close()

    # ── Final evaluation on test + challenge ──────────────────────────────
    if is_main(rank):
        logger.info("Loading best_model2.pt for final evaluation ...")
        raw.load_state_dict(torch.load(ckpt_path, map_location=device))

        results = {
            "lang":          cfg.lang,
            "metric_protocol": cfg.metric_protocol,
            "ribes_script": cfg.ribes_script,
            "variant": {
                "expert_type":     getattr(cfg, "expert_type", "rbf"),
                "n_experts":       cfg.n_experts,
                "use_region_gate": getattr(cfg, "use_region_gate", True),
                "use_aux":         getattr(cfg, "use_aux", True),
                "use_residual":    getattr(cfg, "use_residual", True),
                "raw_coords":      getattr(cfg, "raw_coords", False),
                "seed":            cfg.seed,
                "warmup_steps":    cfg.warmup_steps,
                "patience":        cfg.patience,
                "max_src_len":     cfg.max_src_len,
            },
            "best_dev_bleu": best_bleu,
            "history":       history,
        }

        for split in ["test", "challenge"]:
            split_df = load_split(cfg, split)
            hyps, refs = generate_all(raw, tokenizer, split_df, cfg, device)
            bleu  = compute_bleu(hyps, refs, cfg)
            ribes = compute_ribes(hyps, refs, cfg)
            results[split] = {"BLEU": round(bleu, 2), "RIBES": round(ribes, 4), "n": len(refs)}
            logger.info(f"[{split:>12}]  BLEU={bleu:.2f}  RIBES={ribes:.4f}  (n={len(refs)})")

        out_file = os.path.join(cfg.work_dir, "final_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved → {out_file}")

    cleanup()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the NLLB-based KAN-MoE translation model."
    )
    parser.add_argument(
        "--lang", choices=["hindi", "bengali", "malayalam"], default="hindi",
        help="Target language (default: hindi)",
    )
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume after this epoch (e.g. 7 resumes from epoch 8)")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Checkpoint to load weights from when resuming")
    parser.add_argument("--best-bleu", type=float, default=-1.0,
                        help="Best BLEU achieved before resuming — prevents overwriting a better checkpoint")
    # Expert and component options support controlled comparisons.
    parser.add_argument("--expert", choices=["rbf", "mlp", "spline"], default="rbf",
                        help="Expert type: rbf (full), mlp (param-matched capacity control), spline")
    parser.add_argument("--ablation",
                        choices=["none", "no_region", "no_aux", "raw_coords", "no_residual"],
                        default="none", help="Single-component ablation (Table 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (E1 multi-seed: 42, 1, 2)")
    parser.add_argument("--n-experts", type=int, default=4,
                        help="Number of experts (expert-count control)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the training epoch budget for a controlled experiment")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag; work_dir becomes runs/<lang>_<tag> to avoid overwrite")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing <language>-visual-genome-*.txt files")
    parser.add_argument("--nllb-dir", type=str, default=None,
                        help="Local NLLB-200-1.3B model/tokenizer directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Run output directory (overrides the automatic runs/ path)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML configuration file; its values are applied before CLI overrides")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16",
                        help="Autocast precision (reported experiments use bf16)")
    parser.add_argument("--metric-protocol", choices=["advisor", "legacy"], default="advisor",
                        help="Advisor case-sensitive WAT procedure, or historical local scoring")
    parser.add_argument("--ribes-script", default=None,
                        help="RIBES.py path (advisor protocol requires version 1.02.4)")
    args = parser.parse_args()

    spec = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        args.lang = spec.get("language", args.lang)
        args.data_dir = args.data_dir or spec.get("data_dir")
        args.nllb_dir = args.nllb_dir or spec.get("nllb_dir")
        args.output_dir = args.output_dir or spec.get("output_dir")
        if args.seed is None:
            args.seed = spec.get("seed")
        if args.epochs is None:
            args.epochs = spec.get("epochs")

    data_dir = args.data_dir or _DATA_DIR
    work_dir = args.output_dir
    if work_dir is None:
        work_dir = os.path.join(_HERE, "runs", f"{args.lang}_{args.tag or f'{args.expert}_{args.ablation}_s{args.seed if args.seed is not None else 42}'}")

    # Auto-tag the run so distinct variants/seeds never clobber each other's checkpoints.
    cfg = get_cfg(
        lang=args.lang,
        data_dir=data_dir,
        nllb13b_local=args.nllb_dir or _NLLB13B_DIR,
        work_dir=work_dir,
        ribes_script=args.ribes_script or (_RIBES_ADVISOR if args.metric_protocol == "advisor" else _RIBES),
        indic_nlp_dir=_INDIC_NLP,
    )
    cfg.metric_protocol = args.metric_protocol

    # Apply the selected experiment settings.
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        cfg.epochs = args.epochs
    cfg.expert_type = args.expert
    if args.n_experts < 1:
        parser.error("--n-experts must be at least 1")
    cfg.n_experts = args.n_experts
    cfg.use_region_gate = args.ablation != "no_region"
    cfg.use_aux         = args.ablation != "no_aux"
    cfg.use_residual    = args.ablation != "no_residual"
    cfg.raw_coords      = args.ablation == "raw_coords"
    for key, value in spec.get("model", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    for key, value in spec.get("training", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    for key, value in spec.get("generation", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    # The training loop uses BF16 autocast.
    if args.precision != "bf16":
        parser.error("Only bf16 is implemented by the reference training loop")

    # Validate metric tokenization before launching an expensive training run.
    init_indic_nlp(cfg.indic_nlp_dir, cfg.lang_code)
    validate_metric_setup(cfg)

    resume_ckpt = args.resume_ckpt
    if args.resume_from and resume_ckpt is None:
        resume_ckpt = os.path.join(work_dir, "best_model2.pt")

    run(cfg, resume_from=args.resume_from, resume_ckpt=resume_ckpt, best_bleu_override=args.best_bleu)
