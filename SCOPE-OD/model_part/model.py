from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.alpha = float(alpha)


        self.scaling = self.alpha / max(self.r, 1)


        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None


        if self.r > 0:
            self.lora_a = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, self.r))


            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)


        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        if self.r > 0:

            delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
            out = out + delta

        return out


class MLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        linear = (
            lambda a, b: LoRALinear(a, b, r=lora_rank, alpha=lora_alpha, freeze_base=freeze_base)
        ) if use_lora else (
            lambda a, b: nn.Linear(a, b)
        )

        self.net = nn.Sequential(
            linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UncertaintyAwarePriorEncoder(nn.Module):

    def __init__(self, num_features: int, hidden_dim: int, vocab_size: int = 5):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size


        self.embeddings = nn.Parameter(torch.empty(num_features, vocab_size, hidden_dim))

        nn.init.xavier_uniform_(self.embeddings)


        self.uncertainty_gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hard: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        conf: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hard = hard.long().clamp_min(0).clamp_max(self.vocab_size - 1)


        one_hot = F.one_hot(hard, num_classes=self.vocab_size).float()
        hard_emb = torch.einsum("...fv,fvh->...fh", one_hot, self.embeddings)


        if prob is None:
            prob = one_hot
        soft_emb = torch.einsum("...fv,fvh->...fh", prob.float(), self.embeddings)


        if conf is None:
            conf = torch.ones_like(hard, dtype=torch.float32)
        if entropy is None:
            entropy = torch.zeros_like(hard, dtype=torch.float32)


        gate = self.uncertainty_gate(torch.stack([conf.float(), entropy.float()], dim=-1))


        field_emb = gate * hard_emb + (1.0 - gate) * soft_emb


        pooled = field_emb.mean(dim=-2)


        out = self.out_proj(pooled)
        return self.norm(out + pooled)


class ContextRegimeEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.encoder = MLP(
            input_dim,
            hidden_dim,
            hidden_dim,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class SemanticControlTranslator(nn.Module):

    def __init__(
        self,
        sem_dim: int,
        regime_dim: int,
        pair_static_dim: int,
        cond_dim: int,
        hidden_dim: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        node_in_dim = sem_dim + regime_dim
        pair_in_dim = sem_dim + pair_static_dim + regime_dim


        self.origin_proj = MLP(
            node_in_dim, hidden_dim, cond_dim,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=freeze_base
        )
        self.destination_proj = MLP(
            node_in_dim, hidden_dim, cond_dim,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=freeze_base
        )
        self.pair_gate_proj = MLP(
            pair_in_dim, hidden_dim, 1,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=freeze_base
        )
        self.pair_bias_proj = MLP(
            pair_in_dim, hidden_dim, cond_dim,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=freeze_base
        )
    def forward(
        self,
        origin_sem: torch.Tensor,
        destination_sem: torch.Tensor,
        pair_sem: torch.Tensor,
        pair_static_num: torch.Tensor,
        regime_t: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = origin_sem.shape


        r_node = regime_t.unsqueeze(1).expand(-1, N, -1)
        origin_cond = self.origin_proj(torch.cat([origin_sem, r_node], dim=-1))
        destination_cond = self.destination_proj(torch.cat([destination_sem, r_node], dim=-1))


        r_pair = regime_t.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
        pair_in = torch.cat([pair_sem, pair_static_num, r_pair], dim=-1)


        pair_gate = torch.sigmoid(self.pair_gate_proj(pair_in))
        pair_bias = self.pair_bias_proj(pair_in)


        pair_gate = pair_gate * pair_mask.unsqueeze(-1)
        pair_bias = pair_bias * pair_mask.unsqueeze(-1)

        return origin_cond, destination_cond, pair_gate, pair_bias


class ODConv(nn.Module):

    def __init__(self, K: int, input_dim: int, hidden_dim: int, use_bias: bool = True, activation=None):
        super().__init__()
        self.K = int(K)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation() if activation is not None else None


        self.weight = nn.Parameter(torch.empty(input_dim * (self.K ** 2), hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if use_bias else None

    def _batched_polynomials(self, g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 2:
            g = g.unsqueeze(0)

        B, N, _ = g.shape
        eye = torch.eye(N, device=g.device).unsqueeze(0).expand(B, -1, -1)

        polys = [eye]
        if self.K > 1:
            polys.append(g)

        for k in range(2, self.K):
            polys.append(2 * torch.bmm(g, polys[k - 1]) - polys[k - 2])

        return torch.stack(polys, dim=1)

    def forward(self, X: torch.Tensor, G: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat_set = []


        if isinstance(G, tuple):
            Go, Gd = G

            if Go.dim() == 2 or Go.dim() == 3:
                Go = self._batched_polynomials(Go)
            if Gd.dim() == 2 or Gd.dim() == 3:
                Gd = self._batched_polynomials(Gd)


            for o in range(self.K):
                for d in range(self.K):
                    mode1 = torch.einsum("bncl,bnm->bmcl", X, Go[:, o])
                    mode2 = torch.einsum("bmcl,bcd->bmdl", mode1, Gd[:, d])
                    feat_set.append(mode2)


        else:
            if G.dim() == 3:

                for o in range(self.K):
                    for d in range(self.K):
                        mode1 = torch.einsum("bncl,nm->bmcl", X, G[o])
                        mode2 = torch.einsum("bmcl,cd->bmdl", mode1, G[d])
                        feat_set.append(mode2)
            elif G.dim() == 4:

                for o in range(self.K):
                    for d in range(self.K):
                        mode1 = torch.einsum("bncl,bnm->bmcl", X, G[:, o])
                        mode2 = torch.einsum("bmcl,bcd->bmdl", mode1, G[:, d])
                        feat_set.append(mode2)
            else:
                raise ValueError("graph_kernel must have shape [K, N, N] or [B, K, N, N].")


        feat = torch.cat(feat_set, dim=-1)
        out = torch.einsum("bmdk,kh->bmdh", feat, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return self.activation(out) if self.activation is not None else out


class RoleConditionedFlowMemoryBlock(nn.Module):

    def __init__(
        self,
        K: int,
        input_dim: int,
        hidden_dim: int,
        cond_dim: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim


        self.gates = ODConv(K, input_dim + hidden_dim, hidden_dim * 2)


        self.candi = ODConv(K, input_dim + hidden_dim, hidden_dim)

        linear = (
            lambda a, b: LoRALinear(a, b, r=lora_rank, alpha=lora_alpha, freeze_base=freeze_base)
        ) if use_lora else (
            lambda a, b: nn.Linear(a, b)
        )


        self.origin_gate_film = linear(cond_dim, hidden_dim * 4)
        self.destination_gate_film = linear(cond_dim, hidden_dim * 4)
        self.origin_candi_film = linear(cond_dim, hidden_dim * 2)
        self.destination_candi_film = linear(cond_dim, hidden_dim * 2)


        self.pair_candi_bias = linear(cond_dim, hidden_dim)

    @staticmethod
    def _split_film(x: torch.Tensor, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = torch.split(x, out_dim, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma, beta

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, num_nodes, num_nodes, self.hidden_dim, device=device)

    def forward(
        self,
        G: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        Xt: torch.Tensor,
        Ht_1: torch.Tensor,
        origin_cond: torch.Tensor,
        destination_cond: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> torch.Tensor:

        gate_feat = self.gates(torch.cat([Xt, Ht_1], dim=-1), G)


        og, ob = self._split_film(self.origin_gate_film(origin_cond), self.hidden_dim * 2)
        dg, db = self._split_film(self.destination_gate_film(destination_cond), self.hidden_dim * 2)


        gate_gamma = og.unsqueeze(2) * dg.unsqueeze(1)
        gate_beta = ob.unsqueeze(2) + db.unsqueeze(1)


        gates = gate_feat * gate_gamma + gate_beta

        update_raw, reset_raw = torch.split(gates, self.hidden_dim, dim=-1)
        update = torch.sigmoid(update_raw)
        reset = torch.sigmoid(reset_raw)


        candi_feat = self.candi(torch.cat([Xt, reset * Ht_1], dim=-1), G)

        ocg, ocb = self._split_film(self.origin_candi_film(origin_cond), self.hidden_dim)
        dcg, dcb = self._split_film(self.destination_candi_film(destination_cond), self.hidden_dim)

        candi_gamma = ocg.unsqueeze(2) * dcg.unsqueeze(1)
        candi_beta = ocb.unsqueeze(2) + dcb.unsqueeze(1) + self.pair_candi_bias(pair_bias)

        candi = torch.tanh(candi_feat * candi_gamma + candi_beta)


        Ht = (1.0 - update) * Ht_1 + update * candi
        return Ht


class HistoricalFlowEncoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        K: int,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        cond_dim: int,
        num_layers: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.hidden_dim = [hidden_dim] * num_layers if not isinstance(hidden_dim, list) else hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.num_nodes = num_nodes

        for i in range(num_layers):
            cur_in = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cells.append(
                RoleConditionedFlowMemoryBlock(
                    K,
                    cur_in,
                    self.hidden_dim[i],
                    cond_dim,
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    freeze_base=freeze_base,
                )
            )

    def forward(
        self,
        static_graph: torch.Tensor,
        dynamic_graphs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        X_seq: torch.Tensor,
        origin_conds: Sequence[torch.Tensor],
        destination_conds: Sequence[torch.Tensor],
        pair_biases: Sequence[torch.Tensor],
        H0_l: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        B, T, N, _, _ = X_seq.shape

        if H0_l is None:
            H0_l = [cell.init_hidden(B, N, X_seq.device) for cell in self.cells]

        in_seq = X_seq
        Ht_lst = []

        for l in range(self.num_layers):
            Ht = H0_l[l]
            out_seq = []

            for t in range(T):

                G_cur = dynamic_graphs[t] if l == 0 else static_graph
                Ht = self.cells[l](G_cur, in_seq[:, t], Ht, origin_conds[t], destination_conds[t], pair_biases[t])
                out_seq.append(Ht)

            in_seq = torch.stack(out_seq, dim=1)
            Ht_lst.append(Ht)

        return in_seq, Ht_lst


class FutureFlowDecoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        K: int,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        cond_dim: int,
        num_layers: int,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.hidden_dim = [hidden_dim] * num_layers if not isinstance(hidden_dim, list) else hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            cur_in = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cells.append(
                RoleConditionedFlowMemoryBlock(
                    K,
                    cur_in,
                    self.hidden_dim[i],
                    cond_dim,
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    freeze_base=freeze_base,
                )
            )

    def forward(
        self,
        static_graph: torch.Tensor,
        dynamic_graph: Tuple[torch.Tensor, torch.Tensor],
        Xt: torch.Tensor,
        origin_cond: torch.Tensor,
        destination_cond: torch.Tensor,
        pair_bias: torch.Tensor,
        H0_l: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        Xin = Xt
        Ht_lst = []

        for l in range(self.num_layers):

            G_cur = dynamic_graph if l == 0 else static_graph
            Ht = self.cells[l](G_cur, Xin, H0_l[l], origin_cond, destination_cond, pair_bias)
            Ht_lst.append(Ht)
            Xin = Ht

        return Xin, Ht_lst


class SemanticConsistencyHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_fields: int,
        num_classes: int = 5,
        use_lora: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()

        linear = (
            lambda a, b: LoRALinear(a, b, r=lora_rank, alpha=lora_alpha, freeze_base=freeze_base)
        ) if use_lora else (
            lambda a, b: nn.Linear(a, b)
        )

        self.classifier = linear(hidden_dim, num_fields * num_classes)
        self.num_fields = num_fields
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return logits.view(*x.shape[:-1], self.num_fields, self.num_classes)


class ScopeODNet(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        K: int,
        input_dim: int,
        hidden_dim: int,
        out_horizon: int,
        num_layers: int,
        origin_sem_dim: int,
        destination_sem_dim: int,
        pair_sem_dim: int,
        pair_static_dim: int,
        regime_dim: int,
        semantic_hidden_dim: int = 32,
        semantic_cond_dim: int = 32,
        semantic_vocab_size: int = 5,
        use_regime_token: bool = True,
        use_film_controller: bool = True,
        use_pair_gate: bool = True,
        use_lora_adaptation: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        freeze_base_when_lora: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.out_horizon = out_horizon
        self.use_regime_token = use_regime_token
        self.use_film_controller = use_film_controller
        self.use_pair_gate = use_pair_gate


        self.origin_sem_encoder = UncertaintyAwarePriorEncoder(origin_sem_dim, semantic_hidden_dim, vocab_size=semantic_vocab_size)
        self.destination_sem_encoder = UncertaintyAwarePriorEncoder(destination_sem_dim, semantic_hidden_dim, vocab_size=semantic_vocab_size)
        self.pair_sem_encoder = UncertaintyAwarePriorEncoder(pair_sem_dim, semantic_hidden_dim, vocab_size=semantic_vocab_size)


        self.regime_encoder = ContextRegimeEncoder(
            regime_dim,
            semantic_hidden_dim,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )


        self.controller = SemanticControlTranslator(
            sem_dim=semantic_hidden_dim,
            regime_dim=semantic_hidden_dim,
            pair_static_dim=pair_static_dim,
            cond_dim=semantic_cond_dim,
            hidden_dim=semantic_hidden_dim,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )


        self.encoder = HistoricalFlowEncoder(
            num_nodes,
            K,
            input_dim,
            hidden_dim,
            semantic_cond_dim,
            num_layers,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )
        self.decoder = FutureFlowDecoder(
            num_nodes,
            K,
            input_dim,
            hidden_dim,
            semantic_cond_dim,
            num_layers,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )

        linear = (
            lambda a, b: LoRALinear(a, b, r=lora_rank, alpha=lora_alpha, freeze_base=freeze_base_when_lora)
        ) if use_lora_adaptation else (
            lambda a, b: nn.Linear(a, b)
        )


        self.output_head = linear(hidden_dim, input_dim)
        self.softplus = nn.Softplus()


        self.origin_sem_head = SemanticConsistencyHead(
            hidden_dim,
            origin_sem_dim,
            semantic_vocab_size,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )
        self.destination_sem_head = SemanticConsistencyHead(
            hidden_dim,
            destination_sem_dim,
            semantic_vocab_size,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )
        self.pair_sem_head = SemanticConsistencyHead(
            hidden_dim,
            pair_sem_dim,
            semantic_vocab_size,
            use_lora=use_lora_adaptation,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_when_lora,
        )

    @staticmethod
    def _masked_reduce_pair_to_origin(h: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        w = pair_mask.unsqueeze(-1)
        denom = w.sum(dim=2).clamp_min(1.0)
        return (h * w).sum(dim=2) / denom

    @staticmethod
    def _masked_reduce_pair_to_destination(h: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        w = pair_mask.unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (h * w).sum(dim=1) / denom

    @staticmethod
    def _row_normalize(a: torch.Tensor) -> torch.Tensor:
        deg = a.sum(dim=-1, keepdim=True)
        deg = deg.clamp_min(1e-6)
        return a / deg

    def _build_dynamic_graph(self, base_adj: torch.Tensor, pair_gate: torch.Tensor, pair_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = base_adj.shape
        gate = pair_gate.squeeze(-1)


        if not self.use_pair_gate:
            gate = torch.ones_like(gate)


        adj = base_adj * gate * pair_mask


        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        go = self._row_normalize(adj + eye)
        gd = self._row_normalize(adj.transpose(1, 2) + eye)

        return go, gd

    def _encode_static_semantics(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin_sem = self.origin_sem_encoder(
            batch["origin_city_static_sem"],
            batch.get("origin_city_static_sem_prob"),
            batch.get("origin_city_static_sem_confidence"),
            batch.get("origin_city_static_sem_entropy"),
        )
        destination_sem = self.destination_sem_encoder(
            batch["destination_city_static_sem"],
            batch.get("destination_city_static_sem_prob"),
            batch.get("destination_city_static_sem_confidence"),
            batch.get("destination_city_static_sem_entropy"),
        )
        pair_sem = self.pair_sem_encoder(
            batch["pair_relation_static_sem"],
            batch.get("pair_relation_static_sem_prob"),
            batch.get("pair_relation_static_sem_confidence"),
            batch.get("pair_relation_static_sem_entropy"),
        )
        return origin_sem, destination_sem, pair_sem

    def _build_conditions(
        self,
        origin_sem: torch.Tensor,
        destination_sem: torch.Tensor,
        pair_sem: torch.Tensor,
        pair_static_num: torch.Tensor,
        regime_seq: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[Tuple[torch.Tensor, torch.Tensor]]]:

        if self.use_regime_token:
            regime_latent = self.regime_encoder(regime_seq)
        else:

            regime_latent = torch.zeros(
                (*regime_seq.shape[:-1], origin_sem.shape[-1]),
                device=regime_seq.device
            )

        origin_conds, destination_conds, pair_biases, dynamic_graphs = [], [], [], []

        for t in range(regime_latent.shape[1]):
            o, d, pair_gate, pair_bias = self.controller(
                origin_sem,
                destination_sem,
                pair_sem,
                pair_static_num,
                regime_latent[:, t],
                pair_mask,
            )


            if not self.use_film_controller:
                o = torch.zeros_like(o)
                d = torch.zeros_like(d)
                pair_bias = torch.zeros_like(pair_bias)

            origin_conds.append(o)
            destination_conds.append(d)
            pair_biases.append(pair_bias)


            dynamic_graphs.append(
                self._build_dynamic_graph(
                    base_adj=self._cached_base_adj,
                    pair_gate=pair_gate,
                    pair_mask=pair_mask,
                )
            )

        return origin_conds, destination_conds, pair_biases, dynamic_graphs

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_od = batch["x_od"]
        y_regime = batch["y_regime"]
        x_regime = batch["x_regime"]
        static_graph = batch["graph_kernel"]
        pair_mask = batch["pair_mask"]
        base_adj = batch["base_adj"]
        pair_static_num = batch["pair_static_num"]


        self._cached_base_adj = base_adj


        origin_sem, destination_sem, pair_sem = self._encode_static_semantics(batch)


        x_origin_conds, x_destination_conds, x_pair_biases, x_dynamic_graphs = self._build_conditions(
            origin_sem, destination_sem, pair_sem, pair_static_num, x_regime, pair_mask
        )


        _, Ht_lst = self.encoder(
            static_graph,
            x_dynamic_graphs,
            x_od,
            x_origin_conds,
            x_destination_conds,
            x_pair_biases,
        )


        enc_last = Ht_lst[-1]
        origin_state = self._masked_reduce_pair_to_origin(enc_last, pair_mask)
        destination_state = self._masked_reduce_pair_to_destination(enc_last, pair_mask)
        pair_state = enc_last

        origin_static_logits = self.origin_sem_head(origin_state)
        destination_static_logits = self.destination_sem_head(destination_state)
        pair_static_logits = self.pair_sem_head(pair_state)


        y_origin_conds, y_destination_conds, y_pair_biases, y_dynamic_graphs = self._build_conditions(
            origin_sem, destination_sem, pair_sem, pair_static_num, y_regime, pair_mask
        )


        outputs = []
        decoder_states = []
        cur_states = Ht_lst

        for t in range(self.out_horizon):

            if x_od.shape[1] >= self.out_horizon:
                seasonal_log_ref = x_od[:, -(self.out_horizon - t)]
            else:
                seasonal_log_ref = x_od[:, -1]

            dec_hidden, cur_states = self.decoder(
                static_graph,
                y_dynamic_graphs[t],
                seasonal_log_ref,
                y_origin_conds[t],
                y_destination_conds[t],
                y_pair_biases[t],
                cur_states,
            )


            pred_delta_log = self.output_head(dec_hidden).clamp(min=-2.0, max=2.0)


            pred_log = (seasonal_log_ref + pred_delta_log).clamp_min(0.0)

            pred_raw = torch.expm1(pred_log)

            outputs.append(pred_raw)
            decoder_states.append(dec_hidden)

        pred_raw = torch.stack(outputs, dim=1)
        decoder_states = torch.stack(decoder_states, dim=1)

        return {
            "pred_raw": pred_raw,
            "origin_static_logits": origin_static_logits,
            "destination_static_logits": destination_static_logits,
            "pair_static_logits": pair_static_logits,
            "encoder_last_hidden": enc_last,
            "decoder_hidden": decoder_states,
        }
