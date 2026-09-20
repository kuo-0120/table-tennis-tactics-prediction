# ============================================================
# ShuttleNet-TAA Full Version (pointId collapse fixed)
# - TAA
# - Transition features
# - Weighted focal loss
# - Multi-seed ensemble
# - Logit Adjustment
# - Stochastic Sampling
# - pointId majority suppression
# ============================================================

import argparse
import copy
import os
import random
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

PAD_TOKEN = 0
IGNORE_IDX = -1

FEATURES_SPA = ["pointId", "positionId", "prev_pointId"]
FEATURES_ACT = ["actionId", "spinId", "strikeId", "strengthId", "prev_actionId", "transition_bin"]
FEATURES_CTX = [
    "sex", "handId", "scoreSelf", "scoreOther",
    "scoreDiff_shifted", "scoreSum", "scoreParity",
    "isLeading", "isGamePoint", "isEarlyStrike",
    "isServer", "strikeNumber_clipped", "numberGame",
    "rally_phase", "score_momentum",
]
PLAYER_FEATS = ["gamePlayerId", "gamePlayerOtherId"]
ALL_FEATS_NO_PLAYER = FEATURES_SPA + FEATURES_ACT + FEATURES_CTX
ALL_FEATS_WITH_PLAYER = ALL_FEATS_NO_PLAYER + PLAYER_FEATS

REQUIRED_BASE = [
    "rally_uid","sex","match","numberGame","rally_id","strikeNumber",
    "scoreSelf","scoreOther","gamePlayerId","gamePlayerOtherId",
    "strikeId","handId","strengthId","spinId","pointId","actionId","positionId",
]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transition_matrix(df: pd.DataFrame, n_actions: int) -> np.ndarray:
    mat = np.zeros((n_actions, n_actions), dtype=np.float32)
    for _, g in df.groupby("rally_uid"):
        g = g.sort_values("strikeNumber")
        acts = g["actionId"].astype(int).values
        for i in range(len(acts) - 1):
            a, b = acts[i], acts[i + 1]
            if 0 <= a < n_actions and 0 <= b < n_actions:
                mat[a, b] += 1
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return mat / row_sum


def discretize_transition(prob: float, n_bins: int = 8) -> int:
    return min(int(prob * n_bins) + 1, n_bins)


def preprocess(df: pd.DataFrame, maxlen: int, use_player_id: bool = False,
               trans_mat: Optional[np.ndarray] = None) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_BASE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "serverGetPoint" in df.columns:
        df["serverGetPoint"] = pd.to_numeric(df["serverGetPoint"], errors="coerce").fillna(0).astype(float)
    else:
        df["serverGetPoint"] = 0.0

    df = df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    first_idx = df.groupby("rally_uid")["strikeNumber"].idxmin()
    server_info = df.loc[first_idx, ["rally_uid", "gamePlayerId"]].rename(columns={"gamePlayerId": "_srv"})
    df = df.merge(server_info, on="rally_uid", how="left")
    df["_srv"] = df["_srv"].fillna(0).astype(int)
    df["isServer"] = (df["gamePlayerId"] == df["_srv"]).astype(int)
    df.drop(columns=["_srv"], inplace=True)

    df["scoreDiff_shifted"] = (df["scoreSelf"] - df["scoreOther"] + 50).astype(int)
    df["scoreSum"] = (df["scoreSelf"] + df["scoreOther"]).astype(int)
    df["scoreParity"] = (df["scoreSum"] % 2).astype(int)
    df["isLeading"] = (df["scoreSelf"] > df["scoreOther"]).astype(int)
    df["isGamePoint"] = ((df["scoreSelf"] >= 20) | (df["scoreOther"] >= 20)).astype(int)
    df["isEarlyStrike"] = (df["strikeNumber"] <= 2).astype(int)
    df["strikeNumber_clipped"] = df["strikeNumber"].clip(0, maxlen).astype(int)

    sn = df["strikeNumber"]
    df["rally_phase"] = np.where(sn <= 3, 0, np.where(sn <= 10, 1, 2)).astype(int)

    rolling_diff = df.groupby("rally_uid")["scoreSelf"].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).sum()
    )
    df["score_momentum"] = np.where(rolling_diff > 0, 2, np.where(rolling_diff < 0, 0, 1)).astype(int)

    df["prev_pointId"] = df.groupby("rally_uid")["pointId"].shift(1).fillna(-1).astype(int) + 1
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1).fillna(-1).astype(int) + 1

    if trans_mat is not None:
        probs = np.zeros(len(df), dtype=np.float32)
        prev_a = df["prev_actionId"].values - 1
        curr_a = df["actionId"].values
        n_act = trans_mat.shape[0]
        for i in range(len(df)):
            pa, ca = int(prev_a[i]), int(curr_a[i])
            if 0 <= pa < n_act and 0 <= ca < n_act:
                probs[i] = trans_mat[pa, ca]
        df["transition_bin"] = [discretize_transition(p) for p in probs]
    else:
        df["transition_bin"] = 1

    if not use_player_id:
        df["gamePlayerId"] = 0
        df["gamePlayerOtherId"] = 0

    return df


def build_encoders(train_df, test_df, all_feats):
    both = pd.concat([train_df[all_feats], test_df[all_feats]], ignore_index=True)
    return {c: pd.Categorical(both[c]).categories for c in all_feats}


def encode_frame(df, all_feats, encoders):
    cols = []
    for c in all_feats:
        codes = pd.Categorical(df[c], categories=encoders[c]).codes + 1
        codes = np.where(codes <= 0, PAD_TOKEN, codes).astype(np.int64)
        cols.append(codes)
    return np.stack(cols, axis=1)


def build_target_maps(train_df):
    act_classes = np.sort(train_df["actionId"].dropna().astype(int).unique())
    pt_classes = np.sort(train_df["pointId"].dropna().astype(int).unique())
    act2idx = {int(v): i for i, v in enumerate(act_classes)}
    pt2idx = {int(v): i for i, v in enumerate(pt_classes)}
    return act_classes, pt_classes, act2idx, pt2idx


def build_train_arrays(df, all_feats, encoders, act2idx, pt2idx, maxlen):
    Xs, yAs, yPs, yRs, Ls, UIDs = [], [], [], [], [], []
    for rid, g in df.groupby("rally_uid", sort=True):
        g = g.sort_values("strikeNumber")
        if len(g) < 2:
            continue
        inp = g.iloc[:-1].copy()
        tgt = g.iloc[1:].copy()
        n = min(len(inp), maxlen)
        if n <= 0:
            continue
        x = encode_frame(inp.iloc[:n], all_feats, encoders)
        xpad = np.full((maxlen, len(all_feats)), PAD_TOKEN, dtype=np.int64)
        xpad[:n] = x
        yA = np.full((maxlen,), IGNORE_IDX, dtype=np.int64)
        yP = np.full((maxlen,), IGNORE_IDX, dtype=np.int64)
        yA[:n] = [act2idx[int(v)] for v in tgt["actionId"].iloc[:n].values]
        yP[:n] = [pt2idx[int(v)] for v in tgt["pointId"].iloc[:n].values]
        yr = float(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0.0
        Xs.append(xpad)
        yAs.append(yA)
        yPs.append(yP)
        yRs.append(yr)
        Ls.append(n)
        UIDs.append(int(rid))
    return (
        np.array(Xs, dtype=np.int64),
        np.array(yAs, dtype=np.int64),
        np.array(yPs, dtype=np.int64),
        np.array(yRs, dtype=np.float32),
        np.array(Ls, dtype=np.int64),
        np.array(UIDs, dtype=np.int64),
    )


def build_test_arrays(df, all_feats, encoders, maxlen):
    Xs, Ls, UIDs = [], [], []
    for rid, g in df.groupby("rally_uid", sort=True):
        g = g.sort_values("strikeNumber")
        n = min(len(g), maxlen)
        x = encode_frame(g.iloc[:n], all_feats, encoders)
        xpad = np.full((maxlen, len(all_feats)), PAD_TOKEN, dtype=np.int64)
        xpad[:n] = x
        Xs.append(xpad)
        Ls.append(n)
        UIDs.append(int(rid))
    return np.array(Xs, dtype=np.int64), np.array(Ls, dtype=np.int64), np.array(UIDs, dtype=np.int64)


class RallyDataset(Dataset):
    def __init__(self, X, L, yA=None, yP=None, yR=None, uid=None):
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.L = torch.as_tensor(L, dtype=torch.long)
        self.yA = None if yA is None else torch.as_tensor(yA, dtype=torch.long)
        self.yP = None if yP is None else torch.as_tensor(yP, dtype=torch.long)
        self.yR = None if yR is None else torch.as_tensor(yR, dtype=torch.float32)
        self.uid = None if uid is None else torch.as_tensor(uid, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"X": self.X[idx], "L": self.L[idx]}
        if self.yA is not None:
            item["yA"] = self.yA[idx]
        if self.yP is not None:
            item["yP"] = self.yP[idx]
        if self.yR is not None:
            item["yR"] = self.yR[idx]
        if self.uid is not None:
            item["uid"] = self.uid[idx]
        return item


class TypeAreaAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head ** -0.5
        self.Qs = nn.Linear(dim, dim, bias=False)
        self.Ks = nn.Linear(dim, dim, bias=False)
        self.Vs = nn.Linear(dim, dim, bias=False)
        self.Qa = nn.Linear(dim, dim, bias=False)
        self.Ka = nn.Linear(dim, dim, bias=False)
        self.Va = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, Xs, Xa, mask=None):
        Qs = self._split_heads(self.Qs(Xs))
        Ks = self._split_heads(self.Ks(Xs))
        Vs = self._split_heads(self.Vs(Xs))
        Qa = self._split_heads(self.Qa(Xa))
        Ka = self._split_heads(self.Ka(Xa))
        Va = self._split_heads(self.Va(Xa))

        A = (
            torch.matmul(Qa, Ka.transpose(-2, -1)) +
            torch.matmul(Qa, Ks.transpose(-2, -1)) +
            torch.matmul(Qs, Ka.transpose(-2, -1)) +
            torch.matmul(Qs, Ks.transpose(-2, -1))
        ) * (self.scale / 4.0)

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            A = A.masked_fill(key_mask, float("-inf"))

        A = torch.softmax(A, dim=-1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = self.drop(A)

        out_s = torch.matmul(A, Vs)
        out_a = torch.matmul(A, Va)
        B, H, T, d = out_s.shape
        out = torch.cat([
            out_s.transpose(1, 2).reshape(B, T, H * d),
            out_a.transpose(1, 2).reshape(B, T, H * d)
        ], dim=-1)
        return self.norm(self.out(out) + Xs + Xa)


class ShuttleNetTAA(nn.Module):
    def __init__(self, vocab_sizes: List[int], all_feats: List[str],
                 n_action: int, n_point: int,
                 emb_dim: int = 48, hidden: int = 192,
                 n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.all_feats = all_feats
        self.embs = nn.ModuleList([
            nn.Embedding(v + 1, emb_dim, padding_idx=PAD_TOKEN) for v in vocab_sizes
        ])

        self.spa_idx = [all_feats.index(f) for f in FEATURES_SPA if f in all_feats]
        self.act_idx = [all_feats.index(f) for f in FEATURES_ACT if f in all_feats]
        self.ctx_idx = [all_feats.index(f) for f in FEATURES_CTX if f in all_feats]
        self.ply_idx = [all_feats.index(f) for f in PLAYER_FEATS if f in all_feats]

        spa_dim = len(self.spa_idx) * emb_dim
        act_dim = len(self.act_idx) * emb_dim
        ctx_dim = len(self.ctx_idx) * emb_dim
        ply_dim = len(self.ply_idx) * emb_dim

        self.proj_type = nn.Linear(act_dim, hidden)
        self.proj_area = nn.Linear(spa_dim, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden) if ctx_dim > 0 else None
        self.ply_proj = nn.Linear(ply_dim, hidden) if ply_dim > 0 else None

        self.taa_layers = nn.ModuleList([TypeAreaAttention(hidden, n_heads, dropout) for _ in range(n_layers)])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.player_gru = nn.GRU(hidden, hidden, batch_first=True)

        self.gate = nn.Linear(hidden * 3, hidden * 3)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden)
        )

        self.head_action = nn.Linear(hidden, n_action)
        self.head_point = nn.Linear(hidden + n_action, n_point)
        self.head_win = nn.Linear(hidden + n_action + n_point, 1)
        self.dropout = nn.Dropout(dropout)

    def _get_group_emb(self, X, idx_list):
        if not idx_list:
            return None
        return torch.cat([self.embs[i](X[:, :, i]) for i in idx_list], dim=-1)

    def forward(self, X, L):
        B, T, _ = X.shape
        mask = torch.arange(T, device=X.device)[None, :] >= L[:, None]

        type_emb = self._get_group_emb(X, self.act_idx)
        area_emb = self._get_group_emb(X, self.spa_idx)
        ctx_emb = self._get_group_emb(X, self.ctx_idx)
        ply_emb = self._get_group_emb(X, self.ply_idx)

        ht = self.proj_type(type_emb)
        ha = self.proj_area(area_emb)

        h = ht + ha
        for taa, ff, ln in zip(self.taa_layers, self.ff_layers, self.norms):
            h1 = taa(h, h, mask)
            h = ln(h1 + ff(h1))

        hc = self.ctx_proj(ctx_emb) if (self.ctx_proj is not None and ctx_emb is not None) else torch.zeros_like(h)
        if self.ply_proj is not None and ply_emb is not None:
            hp = self.ply_proj(ply_emb)
            hp, _ = self.player_gru(hp)
        else:
            hp = torch.zeros_like(h)

        fusion_cat = torch.cat([h, hc, hp], dim=-1)
        gate = torch.sigmoid(self.gate(fusion_cat))
        fused = self.fusion(fusion_cat * gate)
        fused = self.dropout(fused)

        act_logits = self.head_action(fused)
        act_probs = act_logits.detach().softmax(dim=-1)
        pt_logits = self.head_point(torch.cat([fused, act_probs], dim=-1))
        pt_probs = pt_logits.detach().softmax(dim=-1)
        win_logits = self.head_win(torch.cat([fused, act_probs, pt_probs], dim=-1)).squeeze(-1)
        return act_logits, pt_logits, win_logits


def focal_loss_weighted(logits, targets, weight=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


def seq_focal_loss_weighted(logits, y, L, weight=None, gamma=2.0):
    B, T, C = logits.shape
    mask = (torch.arange(T, device=logits.device)[None, :] < L[:, None]) & (y != IGNORE_IDX)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    lf = logits[mask]
    yf = y[mask]
    return focal_loss_weighted(lf, yf, weight=weight, gamma=gamma).mean()


def gather_last(logits, L):
    idx = (L.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    return logits.gather(1, idx).squeeze(1)


def class_counts(y, n):
    flat = y.reshape(-1)
    flat = flat[flat >= 0]
    return np.bincount(flat, minlength=n).astype(np.int64) + 1


def effective_num_weights(counts, beta=0.999):
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def logit_adjust(logits, counts, tau):
    if tau <= 0:
        return logits
    prior = torch.tensor(counts, dtype=logits.dtype, device=logits.device)
    prior = prior / prior.sum()
    return logits - tau * torch.log(prior + 1e-12)


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def split_train_val(df, val_size, seed):
    groups = df.groupby("rally_uid")["match"].first().reset_index()
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(groups, groups=groups["match"].astype(str)))
    tr_uids = set(groups.iloc[tr_idx]["rally_uid"])
    va_uids = set(groups.iloc[va_idx]["rally_uid"])
    return df[df["rally_uid"].isin(tr_uids)].copy(), df[df["rally_uid"].isin(va_uids)].copy()


def evaluate(model, loader, device, act_counts, pt_counts, tau_a=0.0, tau_p=0.0):
    model.eval()
    predA, predP, predW = [], [], []
    trueA, trueP, trueW = [], [], []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            L = batch["L"].to(device)
            yA = batch["yA"]
            yP = batch["yP"]
            yR = batch["yR"]

            a, p, w = model(X, L)
            a_last = gather_last(a, L)
            p_last = gather_last(p, L)
            w_last = w.gather(1, (L - 1).view(-1, 1)).squeeze(1)

            a_last = logit_adjust(a_last, act_counts, tau_a)
            p_last = logit_adjust(p_last, pt_counts, tau_p)

            idx = (L.cpu().numpy() - 1)
            yA_last = yA[torch.arange(len(idx)), idx].numpy()
            yP_last = yP[torch.arange(len(idx)), idx].numpy()

            predA.extend(a_last.argmax(-1).cpu().numpy())
            predP.extend(p_last.argmax(-1).cpu().numpy())
            predW.extend(torch.sigmoid(w_last).cpu().numpy())
            trueA.extend(yA_last)
            trueP.extend(yP_last)
            trueW.extend(yR.numpy())

    f1a = f1_score(trueA, predA, average="macro", zero_division=0)
    f1p = f1_score(trueP, predP, average="macro", zero_division=0)
    auc = safe_auc(trueW, predW)
    score = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
    return score, f1a, f1p, auc


def train_one(seed, train_df, val_df, encoders, act_classes, pt_classes, act2idx, pt2idx, args, device):
    print(f"[Seed {seed}] building arrays...")
    Xtr, yAtr, yPtr, yRtr, Ltr, UIDtr = build_train_arrays(train_df, args.all_feats, encoders, act2idx, pt2idx, args.maxlen)
    Xva, yAva, yPva, yRva, Lva, UIDva = build_train_arrays(val_df, args.all_feats, encoders, act2idx, pt2idx, args.maxlen)

    train_ds = RallyDataset(Xtr, Ltr, yAtr, yPtr, yRtr, UIDtr)
    val_ds = RallyDataset(Xva, Lva, yAva, yPva, yRva, UIDva)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    vocab_sizes = [len(encoders[c]) + 1 for c in args.all_feats]
    model = ShuttleNetTAA(
        vocab_sizes, args.all_feats, len(act_classes), len(pt_classes),
        args.emb_dim, args.hidden, args.layers, args.heads, args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    act_counts = class_counts(yAtr, len(act_classes))
    pt_counts = class_counts(yPtr, len(pt_classes))
    act_w = effective_num_weights(act_counts).to(device)
    pt_w = effective_num_weights(pt_counts).to(device)

    best = {"score": -1, "state": None, "tau_a": 0.0, "tau_p": 0.0}

    print(f"[Seed {seed}] train start")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            X = batch["X"].to(device)
            L = batch["L"].to(device)
            yA = batch["yA"].to(device)
            yP = batch["yP"].to(device)
            yR = batch["yR"].to(device)

            a, p, w = model(X, L)
            la = seq_focal_loss_weighted(a, yA, L, weight=act_w, gamma=args.focal_gamma)
            lp = seq_focal_loss_weighted(p, yP, L, weight=pt_w, gamma=args.focal_gamma)
            w_last = w.gather(1, (L - 1).view(-1, 1)).squeeze(1)
            lw = F.binary_cross_entropy_with_logits(w_last, yR)
            loss = 0.4 * la + 0.5 * lp + 0.1 * lw

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        best_local = None
        for tau_a in np.linspace(0, 2.0, 9):
            for tau_p in np.linspace(0, 3.0, 13):
                sc, f1a, f1p, auc = evaluate(model, val_loader, device, act_counts, pt_counts, tau_a, tau_p)
                if best_local is None or sc > best_local[0]:
                    best_local = (sc, f1a, f1p, auc, tau_a, tau_p)

        sc, f1a, f1p, auc, tau_a, tau_p = best_local
        print(f"[Seed {seed}] Epoch {epoch:02d} loss={np.mean(losses):.4f} score={sc:.4f} f1a={f1a:.4f} f1p={f1p:.4f} auc={auc:.4f} tauA={tau_a:.2f} tauP={tau_p:.2f}")

        if sc > best["score"]:
            best["score"] = sc
            best["state"] = copy.deepcopy(model.state_dict())
            best["tau_a"] = tau_a
            best["tau_p"] = tau_p

    model.load_state_dict(best["state"])
    return model, act_counts, pt_counts, best["tau_a"], best["tau_p"]


def predict_ensemble(models_info, test_df, encoders, act_classes, pt_classes, args, device):
    Xte, Lte, UIDte = build_test_arrays(test_df, args.all_feats, encoders, args.maxlen)
    test_ds = RallyDataset(Xte, Lte, uid=UIDte)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    all_a = None
    all_p = None
    all_w = None

    for info in models_info:
        model = info["model"]
        model.eval()
        fold_a, fold_p, fold_w = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                X = batch["X"].to(device)
                L = batch["L"].to(device)
                a, p, w = model(X, L)
                a_last = gather_last(a, L)
                p_last = gather_last(p, L)
                w_last = w.gather(1, (L - 1).view(-1, 1)).squeeze(1)

                a_last = logit_adjust(a_last, info["act_counts"], info["tau_a"])
                p_last = logit_adjust(p_last, info["pt_counts"], info["tau_p"])

                fold_a.append(torch.softmax(a_last, dim=-1).cpu())
                fold_p.append(torch.softmax(p_last, dim=-1).cpu())
                fold_w.append(torch.sigmoid(w_last).cpu())

        fold_a = torch.cat(fold_a, dim=0)
        fold_p = torch.cat(fold_p, dim=0)
        fold_w = torch.cat(fold_w, dim=0)

        all_a = fold_a if all_a is None else all_a + fold_a
        all_p = fold_p if all_p is None else all_p + fold_p
        all_w = fold_w if all_w is None else all_w + fold_w

    all_a /= len(models_info)
    all_p /= len(models_info)
    all_w /= len(models_info)

    rng = np.random.default_rng(0)
    rows = []
    for i, uid in enumerate(UIDte):
        a_prob = all_a[i].numpy()
        p_prob = all_p[i].numpy()

        p_prob_adj = p_prob.copy()
        if len(p_prob_adj) > 0:
            p_prob_adj[0] *= args.point0_suppress
            s = p_prob_adj.sum()
            if s > 0:
                p_prob_adj = p_prob_adj / s
            else:
                p_prob_adj = p_prob

        a_samples = rng.choice(len(act_classes), size=args.n_samples, p=a_prob)
        p_samples = rng.choice(len(pt_classes), size=args.n_samples, p=p_prob_adj)

        a_pred = int(np.bincount(a_samples).argmax())
        p_pred = int(np.bincount(p_samples).argmax())

        rows.append({
            "rally_uid": int(uid),
            "actionId": int(act_classes[a_pred]),
            "pointId": int(pt_classes[p_pred]),
            "serverGetPoint": float(np.clip(all_w[i].item(), 1e-6, 1 - 1e-6)),
        })

    return pd.DataFrame(rows).sort_values("rally_uid").reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--sample", default="sample_submission.csv")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--maxlen", type=int, default=70)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 520, 2026])
    p.add_argument("--emb_dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--point0_suppress", type=float, default=0.3)
    p.add_argument("--use_player_id", action="store_true")
    return p.parse_args()


def main():
    print("STARTING FULL TAA...")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)

    if not os.path.exists(args.train):
        print("train file not found:", args.train)
        return
    if not os.path.exists(args.test):
        print("test file not found:", args.test)
        return

    print("reading csv...")
    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    print("train shape =", train_raw.shape)
    print("test shape =", test_raw.shape)

    n_actions = int(train_raw["actionId"].max()) + 1
    print("building transition matrix...")
    trans_mat = build_transition_matrix(train_raw, n_actions)

    print("preprocessing...")
    train_df = preprocess(train_raw, args.maxlen, args.use_player_id, trans_mat)
    test_df = preprocess(test_raw, args.maxlen, args.use_player_id, None)

    args.all_feats = ALL_FEATS_WITH_PLAYER if args.use_player_id else ALL_FEATS_NO_PLAYER

    print("building encoders...")
    encoders = build_encoders(train_df, test_df, args.all_feats)
    act_classes, pt_classes, act2idx, pt2idx = build_target_maps(train_df)

    models_info = []
    for seed in args.seeds:
        seed_everything(seed)
        print("=" * 70)
        print("SEED =", seed)
        tr_df, va_df = split_train_val(train_df, args.val_size, seed)
        print("train rallies =", tr_df["rally_uid"].nunique(), "val rallies =", va_df["rally_uid"].nunique())

        model, act_counts, pt_counts, tau_a, tau_p = train_one(
            seed, tr_df, va_df, encoders, act_classes, pt_classes, act2idx, pt2idx, args, device
        )
        models_info.append({
            "model": model,
            "act_counts": act_counts,
            "pt_counts": pt_counts,
            "tau_a": tau_a,
            "tau_p": tau_p,
        })

    print("predicting ensemble...")
    pred_df = predict_ensemble(models_info, test_df, encoders, act_classes, pt_classes, args, device)

    print("pred rows =", len(pred_df))
    print(pred_df.head())
    print(pred_df["pointId"].value_counts().head(20))

    pred_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("saved to", args.out)


if __name__ == "__main__":
    main()