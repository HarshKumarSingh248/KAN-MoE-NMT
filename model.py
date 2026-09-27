
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from config import CFG


class KANExpert(nn.Module):
    """
    Radial-basis KAN expert with learned centers and bandwidths.
    """
    def __init__(self, d_in: int, d_out: int, n_basis: int, dropout: float = 0.1):
        super().__init__()
        self.d_in    = d_in
        self.n_basis = n_basis
        self.centers   = nn.Parameter(torch.randn(n_basis, d_in) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(n_basis))
        self.out_proj  = nn.Linear(n_basis * d_in, d_out, bias=True)
        self.dropout   = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        diff  = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        sigma = self.log_sigma.exp().clamp(min=1e-4)
        phi   = torch.exp(-0.5 * (diff / sigma.view(1, 1, -1, 1)) ** 2)
        out   = phi.reshape(B, T, self.n_basis * D)
        return self.out_proj(self.dropout(out))


class MLPExpert(nn.Module):
    """
    Parameter-matched MLP expert for capacity-control experiments.

    2-layer MLP: Linear(d_in -> hidden) -> GELU -> Dropout -> Linear(hidden -> d_out).
    The default hidden is chosen so total params match the RBF KANExpert to within ~0.03%:
      KANExpert(d_in=2048, n_basis=16) = 67.14M params (99.9% in out_proj Linear(16*2048->2048)).
      MLP with hidden=16393 = 67.16M params. Pass mlp_hidden=None to auto-match.
    """
    def __init__(self, d_in: int, d_out: int, n_basis: int,
                 dropout: float = 0.1, mlp_hidden: int | None = None):
        super().__init__()
        if mlp_hidden is None:
            # Match KANExpert param count: centers(n_basis*d_in) + n_basis
            #  + out_proj((n_basis*d_in)*d_out + d_out). Solve 2*d_in*h ≈ kan_params.
            kan_params = n_basis * d_in + n_basis + (n_basis * d_in) * d_out + d_out
            mlp_hidden = round(kan_params / (2 * d_in))
        self.fc1     = nn.Linear(d_in, mlp_hidden)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(mlp_hidden, d_out)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


def _b_spline_bases(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Cox-de Boor recursion for B-spline basis functions of order k+1.
    x:    (N, D)
    grid: (D, grid_size+1+2k)   extended grid, k extra knots each side
    Returns: (N, D, grid_size+k) basis values per input dimension.
    """
    x = x.unsqueeze(-1)                                          # (N, D, 1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # (N, D, grid_size+2k)
    for r in range(1, k + 1):
        left_den  = (grid[:, r:-1] - grid[:, :-(r + 1)]).clamp(min=1e-6)
        right_den = (grid[:, r + 1:] - grid[:, 1:-r]).clamp(min=1e-6)
        left  = (x - grid[:, :-(r + 1)]) / left_den * bases[:, :, :-1]
        right = (grid[:, r + 1:] - x) / right_den * bases[:, :, 1:]
        bases = left + right
    return bases                                                 # (N, D, grid_size+k)


class SplineKANExpert(nn.Module):
    """
    B-spline KAN expert using Cox-de Boor recursion with order k=3.

    grid_size is chosen so that (grid_size + k) == n_basis exactly, matching the
    KANExpert out_proj input width (n_basis * d_in) for a fair parameter comparison:
        grid_size = n_basis - k = 16 - 3 = 13  →  67.11M params/expert (RBF: 67.14M, -0.049%).

    A naive Cox-de Boor recursion is tractable at this scale (d_in=2048, grid=13,
    k=3): the intermediate (B,T,D,grid_size+k) tensor is ~0.1GB in bf16, well within
    budget, so no C++/CUDA extension (as used by the `efficient-kan` package) is needed.
    """
    def __init__(self, d_in: int, d_out: int, n_basis: int,
                 dropout: float = 0.1, k: int = 3,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.d_in = d_in
        self.k = k
        self.grid_size = n_basis - k               # 13 for n_basis=16, k=3
        self.n_basis_out = self.grid_size + k       # == n_basis (16)
        step = (grid_range[1] - grid_range[0]) / self.grid_size
        inner = torch.linspace(grid_range[0], grid_range[1], self.grid_size + 1)
        full = torch.cat([
            inner[0] - step * torch.arange(k, 0, -1),
            inner,
            inner[-1] + step * torch.arange(1, k + 1),
        ])                                           # (grid_size+1+2k,)
        self.register_buffer("grid", full.unsqueeze(0).expand(d_in, -1).contiguous())
        # Single out_proj (matches KANExpert's out_proj shape exactly: n_basis*d_in -> d_out,
        # with bias) so total params equal KANExpert to within rounding.
        self.out_proj = nn.Linear(d_in * self.n_basis_out, d_out, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # SiLU base residual (per-dim), as in efficient-kan; folded into the basis
        # features (added to the first basis column) before the shared out_proj,
        # so no extra d_in x d_out projection is introduced (keeps param count exact).
        self.base_weight = nn.Parameter(torch.zeros(d_in))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xf = x.reshape(B * T, D)
        lo = self.grid[:, self.k].min()
        hi = self.grid[:, -(self.k + 1)].max()
        xc = xf.clamp(lo.item(), hi.item())
        bases = _b_spline_bases(xc, self.grid, self.k)           # (N, D, grid_size+k)
        base = F.silu(xf) * self.base_weight                     # (N, D) SiLU residual
        bases = bases.clone()
        bases[..., 0] = bases[..., 0] + base                     # fold into first basis column
        feat = bases.reshape(B * T, D * self.n_basis_out)
        return self.out_proj(self.dropout(feat)).reshape(B, T, -1)


def _make_expert(expert_type: str, d_in: int, d_out: int, n_basis: int, dropout: float):
    if expert_type == "rbf":
        return KANExpert(d_in, d_out, n_basis, dropout)
    if expert_type == "mlp":
        return MLPExpert(d_in, d_out, n_basis, dropout)
    if expert_type == "spline":
        return SplineKANExpert(d_in, d_out, n_basis, dropout)
    raise ValueError(f"Unknown expert_type: {expert_type!r} (use rbf|mlp|spline)")


class KANMoEFusion(nn.Module):
    """
    Softly routes token representations across KAN experts.

    The Switch-style load-balancing term is n_experts times the squared
    L2 norm of the mean routing probabilities. Optional configuration
    settings support expert and component ablations.
    """
    def __init__(self, d_model: int, kan_hidden: int, n_experts: int,
                 n_basis: int, dropout: float = 0.1,
                 expert_type: str = "rbf", use_aux: bool = True,
                 use_residual: bool = True):
        super().__init__()
        self.n_experts    = n_experts
        self.expert_type  = expert_type
        self.use_aux      = use_aux
        self.use_residual = use_residual
        self.experts   = nn.ModuleList([
            _make_expert(expert_type, kan_hidden, kan_hidden, n_basis, dropout)
            for _ in range(n_experts)
        ])
        self.in_proj  = nn.Linear(d_model, kan_hidden)
        self.gate     = nn.Linear(kan_hidden, n_experts)
        self.out_proj = nn.Linear(kan_hidden, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.last_gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.in_proj(x)                                    # (B, T, kan_hidden)
        gates = F.softmax(self.gate(h), dim=-1)                # (B, T, n_experts)
        self.last_gates = gates.detach().cpu()

        expert_outs = torch.stack([e(h) for e in self.experts], dim=2)
        mixed = (gates.unsqueeze(-1) * expert_outs).sum(dim=2) # (B, T, kan_hidden)

        # Correct Switch aux loss: L = n_experts * sum(f_bar^2)
        if self.use_aux:
            f_bar    = gates.mean(dim=[0, 1])                   # (n_experts,)
            aux_loss = self.n_experts * (f_bar ** 2).sum()
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        out = self.out_proj(self.dropout(mixed))                # (B, T, d_model)
        if self.use_residual:
            return self.norm(x + out), aux_loss                 # residual + norm
        return self.norm(out), aux_loss                         # ablation: no skip


class RegionGate(nn.Module):
    """
    Maps bbox Fourier features to an additive bias on encoder states.
    """
    def __init__(self, d_model: int, n_freqs: int = 16, raw_coords: bool = False):
        super().__init__()
        self.raw_coords = raw_coords                           # w/o Fourier ablation
        in_dim = 4 if raw_coords else 4 * 2 * n_freqs          # 4 raw, else 128
        self.gate_proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.n_freqs = n_freqs

    def _fourier(self, bbox: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(4):
            u = bbox[:, i]
            for k in range(self.n_freqs):
                s = 2.0 ** k
                feats.append(torch.sin(s * torch.pi * u))
                feats.append(torch.cos(s * torch.pi * u))
        return torch.stack(feats, dim=-1)                       # (B, 128)

    def forward(self, h: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        phi  = bbox if self.raw_coords else self._fourier(bbox)  # (B,4) or (B,128)
        bias = self.gate_proj(phi).unsqueeze(1)                 # (B, 1, d_model)
        return self.norm(h + bias)                              # additive residual


class KANMoEModel(nn.Module):
    """
    NLLB-1.3B translation model with RegionGate and KAN-MoE.

    Pipeline:
      src_text → NLLB-1.3B encoder → h_enc
      h_enc    → RegionGate(bbox)  → h_spatial   (additive Fourier bias first)
      h_spatial → KAN-MoE        → h_fused     (residual fusion, auxiliary loss)
      h_fused  → NLLB-1.3B decoder → translation

    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        self.nllb13b = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb13b_local)

        # Optional expert and component ablation settings.
        self.expert_type     = getattr(cfg, "expert_type", "rbf")
        self.use_region_gate = getattr(cfg, "use_region_gate", True)  # w/o RegionGate ablation
        self.raw_coords      = getattr(cfg, "raw_coords", False)      # w/o Fourier ablation

        kan_hidden = cfg.kan_hidden
        self.region_gate = RegionGate(
            d_model=cfg.d_model, n_freqs=getattr(cfg, "fourier_freqs", 16),
            raw_coords=self.raw_coords,
        )
        self.kan_moe = KANMoEFusion(
            d_model=cfg.d_model, kan_hidden=kan_hidden,
            n_experts=cfg.n_experts, n_basis=cfg.kan_basis,
            dropout=cfg.kan_dropout,
            expert_type=self.expert_type,
            use_aux=getattr(cfg, "use_aux", True),          # w/o Aux Loss ablation
            use_residual=getattr(cfg, "use_residual", True),  # w/o Residual ablation
        )

    def _build_encoder_states(self, input_ids, attention_mask, bbox):
        enc_out = self.nllb13b.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        h_enc     = enc_out.last_hidden_state
        if self.use_region_gate:
            h_spatial = self.region_gate(h_enc, bbox)  # spatial first
        else:
            h_spatial = h_enc                          # w/o RegionGate ablation
        h_fused, aux_loss = self.kan_moe(h_spatial)    # then KAN-MoE with residual
        return h_fused, attention_mask, aux_loss

    def forward(self, input_ids, attention_mask, labels, bbox,
                label_smoothing=0.1, **kwargs):
        h_enc, mask_enc, aux_loss = self._build_encoder_states(
            input_ids, attention_mask, bbox,
        )
        out = self.nllb13b(
            input_ids=input_ids,
            attention_mask=mask_enc,
            labels=labels,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
        )

        if label_smoothing > 0:
            logits    = out.logits
            V         = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            mask      = labels != -100
            nll    = -log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.sum(dim=-1) / V
            loss   = ((1 - label_smoothing) * nll + label_smoothing * smooth)
            loss   = loss[mask].mean()
        else:
            loss = out.loss

        return loss + self.cfg.moe_aux_wt * aux_loss, aux_loss.detach()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, bbox, tokenizer, **kwargs):
        h_enc, mask_enc, _ = self._build_encoder_states(input_ids, attention_mask, bbox)
        forced_bos = tokenizer.convert_tokens_to_ids(self.cfg.tgt_lang_nllb)
        out_ids = self.nllb13b.generate(
            input_ids=input_ids,
            attention_mask=mask_enc,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
            forced_bos_token_id=forced_bos,
            max_new_tokens=self.cfg.max_tgt_len,
            num_beams=self.cfg.beam_size,
            length_penalty=self.cfg.length_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    def get_param_groups(self):
        seen = set()
        def dedup(params):
            out = []
            for p in params:
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
            return out

        new_params = dedup(
            list(self.kan_moe.parameters()) +
            list(self.region_gate.parameters())
        )
        decoder_params = dedup(
            list(self.nllb13b.model.decoder.parameters()) +
            list(self.nllb13b.lm_head.parameters())
        )
        encoder_params = dedup(
            list(self.nllb13b.model.encoder.parameters())
        )
        return [
            {"params": new_params,     "lr": self.cfg.lr_kan,     "name": "kan_moe_region_gate"},
            {"params": decoder_params, "lr": self.cfg.lr_decoder, "name": "decoder"},
            {"params": encoder_params, "lr": self.cfg.lr_encoder, "name": "encoder"},
        ]

    def unfreeze_encoder_layers(self, from_top: int, to_top: int, lr: float):
        """
        Unfreeze encoder layers[-to_top : -from_top] (exclusive of already-unfrozen top-from_top).
        Call with from_top=0, to_top=4 first; then from_top=4, to_top=8 next.
        This guarantees zero parameter overlap between successive calls.
        """
        layers = self.nllb13b.model.encoder.layers
        if from_top == 0:
            target_layers = layers[-to_top:]
        else:
            target_layers = layers[-to_top:-from_top]
        params = []
        for layer in target_layers:
            for p in layer.parameters():
                p.requires_grad_(True)
                params.append(p)
        return {"params": params, "lr": lr, "name": f"nllb13b_enc_top{to_top}"}

    def param_counts(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
