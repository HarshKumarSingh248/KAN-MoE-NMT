"""Generate translations and compute the repository's BLEU and RIBES metrics."""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers

from config import get_cfg
from data_utils import HVGDataset, compute_bleu, compute_ribes, init_indic_nlp, validate_metric_setup, load_split, make_collate_fn
from model import KANMoEModel

transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")
ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def generate_all(model, tokenizer, frame, cfg, device, batch_size):
    model.eval()
    dataset = HVGDataset(frame)
    collate = make_collate_fn(tokenizer, cfg.src_lang_nllb, cfg.tgt_lang_nllb,
                              cfg.max_src_len, cfg.max_tgt_len)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate,
                        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    hypotheses, references = [], []
    for batch in loader:
        outputs = model.generate(batch["input_ids"].to(device),
                                 batch["attention_mask"].to(device),
                                 batch["bbox"].to(device), tokenizer)
        hypotheses.extend(outputs)
        references.extend(batch["tgt_texts"])
    return hypotheses, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Language YAML config")
    parser.add_argument("--checkpoint", required=True, help="KAN-MoE model state-dict checkpoint")
    parser.add_argument("--split", required=True, choices=["dev", "test", "challenge"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--nllb-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric-protocol", choices=["advisor", "legacy"], default="advisor",
                        help="Advisor case-sensitive WAT procedure, or historical local scoring")
    parser.add_argument("--ribes-script", default=None,
                        help="RIBES.py path (advisor protocol requires version 1.02.4)")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lang = spec["language"]
    data_dir = args.data_dir or spec.get("data_dir", str(ROOT / "data" / lang))
    nllb_dir = args.nllb_dir or spec.get("nllb_dir", str(ROOT / "models" / "nllb-200-1.3B"))
    out_dir = Path(args.output_dir or spec.get("output_dir", str(ROOT / "runs" / lang)))
    ribes_script = args.ribes_script or str(
        ROOT / "third_party" / "RIBES-1.02.4" / "RIBES.py"
        if args.metric_protocol == "advisor" else ROOT / "RIBES.py"
    )
    cfg = get_cfg(lang, data_dir, nllb_dir, str(out_dir), ribes_script, str(ROOT / "third_party" / "indic_nlp_library"))
    cfg.metric_protocol = args.metric_protocol
    for group in ("model", "training", "generation"):
        for key, value in spec.get(group, {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not Path(nllb_dir, "config.json").is_file():
        parser.error(f"NLLB model files not found under {nllb_dir}")
    init_indic_nlp(cfg.indic_nlp_dir, cfg.lang_code)
    validate_metric_setup(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(nllb_dir, local_files_only=True)
    tokenizer.src_lang = cfg.src_lang_nllb
    tokenizer.tgt_lang = cfg.tgt_lang_nllb
    model = KANMoEModel(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    frame = load_split(cfg, args.split)
    hyps, refs = generate_all(model, tokenizer, frame, cfg, device, args.batch_size)
    bleu = compute_bleu(hyps, refs, cfg)
    ribes = compute_ribes(hyps, refs, cfg)
    result = {"language": lang, "split": args.split, "checkpoint": str(checkpoint),
              "metric_protocol": cfg.metric_protocol, "ribes_script": cfg.ribes_script,
              "n": len(refs), "BLEU": round(bleu, 2), "RIBES": round(ribes, 4)}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"evaluation_{lang}_{args.split}_{cfg.metric_protocol}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("%s %s: BLEU %.2f | RIBES %.4f | n=%d", lang, args.split, bleu, ribes, len(refs))
    logger.info("Computed evaluation saved to %s", out_path)


if __name__ == "__main__":
    main()
