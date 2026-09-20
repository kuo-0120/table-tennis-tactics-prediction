import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 0. Global config
# ==============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PAD_TOKEN = 0
MAX_LEN = 40
NUM_FOLDS = 5
EARLY_STOP_PATIENCE = 4

FEATURES_SPA = [
    "pointId",
    "positionId",
    "prev_pointId",
    "courtHalf",
    "sameSideAsPrev",
]

FEATURES_ACT = [
    "actionId",
    "spinId",
    "strikeId",
    "prev_actionId",
]

FEATURES_CTX = [
    "sex",
    "handId",
    "strengthId",
    "scoreSelf",
    "scoreOther",
    "scoreDiff",
    "scoreSum",
    "isGamePoint",
    "scoreParity",
    "isLeading",
    "gamePlayerId",
    "gamePlayerOtherId",
    "isServer",
    "isServe",
    "gameScorePhase",
    "isDeuce",
    "isOvertime",
    "strikeNumber",
]

ALL_FEATURES = FEATURES_SPA + FEATURES_ACT + FEATURES_CTX
FEATURE_IDX = {feat: idx for idx, feat in enumerate(ALL_FEATURES)}


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]


class MaskedWeightedCE(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.03):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, mask):
        C = inputs.shape[-1]
        inputs_flat = inputs.reshape(-1, C)[mask.reshape(-1)]
        targets_flat = targets.reshape(-1)[mask.reshape(-1)]

        if targets_flat.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=inputs.device)

        return F.cross_entropy(
            inputs_flat,
            targets_flat,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class ShuttleNetFusion(nn.Module):
    def __init__(self, vocab_sizes, act_classes_num, pt_classes_num,
                 emb_dim=48, hidden_dim=160, num_layers=2):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_sizes.get(feat, 500) + 2,
                               emb_dim,
                               padding_idx=PAD_TOKEN)
            for feat in ALL_FEATURES
        })

        dim_spa = len(FEATURES_SPA) * emb_dim
        dim_act = len(FEATURES_ACT) * emb_dim
        dim_ctx = len(FEATURES_CTX) * emb_dim

        self.rnn_spa = nn.GRU(
            dim_spa, hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=0.5 if num_layers > 1 else 0.0,
        )
        self.rnn_act = nn.GRU(
            dim_act, hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=0.5 if num_layers > 1 else 0.0,
        )

        spa_act_dim = hidden_dim * 2
        self.spa_act_proj = nn.Linear(spa_act_dim, spa_act_dim)
        self.ctx_proj = nn.Linear(dim_ctx, spa_act_dim)
        self.gate_layer = nn.Linear(spa_act_dim * 2, spa_act_dim)

        fusion_in_dim = spa_act_dim + dim_ctx
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim),
            nn.LayerNorm(fusion_in_dim),
            nn.GELU(),
            nn.Dropout(0.35),
        )

        wide_dim = fusion_in_dim + dim_ctx

        self.head_act = nn.Sequential(
            nn.Linear(wide_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, act_classes_num),
        )

        self.head_pt = nn.Sequential(
            nn.Linear(wide_dim + act_classes_num, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, pt_classes_num),
        )

        self.head_win = nn.Sequential(
            nn.Linear(wide_dim + act_classes_num + pt_classes_num,
                      hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X, lengths):
        emb_spa = torch.cat(
            [self.embeddings[feat](X[:, :, FEATURE_IDX[feat]])
             for feat in FEATURES_SPA],
            dim=-1,
        )
        emb_act = torch.cat(
            [self.embeddings[feat](X[:, :, FEATURE_IDX[feat]])
             for feat in FEATURES_ACT],
            dim=-1,
        )
        emb_ctx = torch.cat(
            [self.embeddings[feat](X[:, :, FEATURE_IDX[feat]])
             for feat in FEATURES_CTX],
            dim=-1,
        )

        out_spa, _ = self.rnn_spa(emb_spa)
        out_act, _ = self.rnn_act(emb_act)

        spa_act = torch.cat([out_spa, out_act], dim=-1)
        spa_act_p = self.spa_act_proj(spa_act)
        ctx_p = self.ctx_proj(emb_ctx)
        gate_in = torch.cat([spa_act_p, ctx_p], dim=-1)
        gate = torch.sigmoid(self.gate_layer(gate_in))
        fused_core = gate * spa_act_p + (1.0 - gate) * ctx_p

        fusion_in = torch.cat([fused_core, emb_ctx], dim=-1)
        final_repr = self.fusion(fusion_in)

        wide_deep_repr = torch.cat([final_repr, emb_ctx], dim=-1)

        pred_act = self.head_act(wide_deep_repr)
        act_probs = F.softmax(pred_act, dim=-1)

        pt_input = torch.cat([wide_deep_repr, act_probs], dim=-1)
        pred_pt = self.head_pt(pt_input)
        pt_probs = F.softmax(pred_pt, dim=-1)

        win_input = torch.cat([wide_deep_repr, act_probs, pt_probs], dim=-1)
        pred_win = self.head_win(win_input).squeeze(-1)

        return pred_act, pred_pt, pred_win


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rally_uid" not in df.columns and "rallyuid" in df.columns:
        df = df.rename(columns={"rallyuid": "rally_uid"})

    df["gamePlayerId"] = df["gamePlayerId"].fillna(0).astype(int)
    df["gamePlayerOtherId"] = df["gamePlayerOtherId"].fillna(0).astype(int)

    df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"] + 30
    df["scoreSum"] = df["scoreSelf"] + df["scoreOther"]
    df["isGamePoint"] = (
        (df["scoreSelf"] >= 20) | (df["scoreOther"] >= 20)
    ).astype(int)
    df["scoreParity"] = df["scoreSum"] % 2
    df["isLeading"] = (df["scoreSelf"] > df["scoreOther"]).astype(int)

    max_score = df[["scoreSelf", "scoreOther"]].max(axis=1)
    phase = np.zeros(len(df), dtype=np.int64)
    phase[(max_score >= 6) & (max_score <= 15)] = 1
    phase[max_score > 15] = 2
    df["gameScorePhase"] = phase

    deuce_cond = (df["scoreSelf"] >= 20) & (df["scoreOther"] >= 20)
    df["isDeuce"] = (deuce_cond & (df["scoreSelf"] == df["scoreOther"])).astype(int)
    df["isOvertime"] = (deuce_cond & (df["scoreSelf"] != df["scoreOther"])).astype(int)

    df = df.sort_values(["rally_uid", "strikeNumber"])
    min_strike_idx = df.groupby("rally_uid")["strikeNumber"].idxmin()
    server_info = df.loc[min_strike_idx, ["rally_uid", "gamePlayerId"]].rename(
        columns={"gamePlayerId": "serverPlayerId"}
    )
    df = df.merge(server_info, on="rally_uid", how="left")
    df["isServer"] = (df["gamePlayerId"] == df["serverPlayerId"]).astype(int)

    df["isServe"] = 0
    df.loc[min_strike_idx, "isServe"] = 1

    pos = df["positionId"].fillna(0).astype(int)
    court_half = np.zeros(len(df), dtype=np.int64)
    court_half[(pos >= 60) & (pos < 120)] = 1
    court_half[pos >= 120] = 2
    df["courtHalf"] = court_half

    df["prev_courtHalf"] = (
        df.groupby("rally_uid")["courtHalf"].shift(1).fillna(-1).astype(int)
    )
    df["sameSideAsPrev"] = (
        (df["courtHalf"] == df["prev_courtHalf"]) & (df["prev_courtHalf"] >= 0)
    ).astype(int)

    df["prev_pointId"] = (
        df.groupby("rally_uid")["pointId"].shift(1).fillna(-1).astype(int) + 1
    )
    df["prev_actionId"] = (
        df.groupby("rally_uid")["actionId"].shift(1).fillna(-1).astype(int) + 1
    )

    return df


def build_sequences(df: pd.DataFrame, is_train: bool = True):
    grouped = df.groupby("rally_uid")

    X_list, yA_list, yP_list, yR_list, L_list, uids = [], [], [], [], [], []
    num_features = len(ALL_FEATURES)

    for rid, group in grouped:
        group = group.sort_values("strikeNumber")
        seq_len = len(group)
        if seq_len < 2 and is_train:
            continue

        effective_len = min(seq_len, MAX_LEN)
        L_list.append(effective_len)
        uids.append(rid)

        x_seq = np.zeros((MAX_LEN, num_features), dtype=np.int64)
        for idx, feat in enumerate(ALL_FEATURES):
            vals = group[feat].values.astype(np.int64)
            length = min(len(vals), MAX_LEN)
            x_seq[:length, idx] = vals[:length]
        X_list.append(x_seq)

        if is_train:
            ya_seq = np.zeros(MAX_LEN, dtype=np.int64)
            yp_seq = np.zeros(MAX_LEN, dtype=np.int64)

            acts = group["actionId"].values[1:]
            pts = group["pointId"].values[1:]
            length = min(len(acts), MAX_LEN)
            if length > 0:
                ya_seq[:length] = acts[:length]
                yp_seq[:length] = pts[:length]

            yA_list.append(ya_seq)
            yP_list.append(yp_seq)
            yR_list.append(group["serverGetPoint"].values[0])

    if is_train:
        return (
            np.array(X_list),
            np.array(yA_list),
            np.array(yP_list),
            np.array(yR_list, dtype=np.float32),
            np.array(L_list, dtype=np.int64),
            np.array(uids),
        )
    else:
        return (
            np.array(X_list),
            np.array(L_list, dtype=np.int64),
            uids,
        )


def evaluate_model(model, val_loader, device):
    model.eval()
    true_act, pred_act_list = [], []
    true_pt, pred_pt_list = [], []
    true_win, pred_win_list = [], []

    with torch.no_grad():
        for X, yA, yP, yR, L in val_loader:
            X = X.to(device)
            yA = yA.to(device)
            yP = yP.to(device)
            yR = yR.to(device)
            L = L.to(device)

            pred_act, pred_pt, pred_win = model(X, L)

            for i in range(len(X)):
                valid_l = L[i].item() - 1
                if valid_l > 0:
                    true_act.extend(yA[i, :valid_l].cpu().numpy())
                    pred_act_list.extend(
                        torch.argmax(pred_act[i, :valid_l], dim=-1).cpu().numpy()
                    )
                    true_pt.extend(yP[i, :valid_l].cpu().numpy())
                    pred_pt_list.extend(
                        torch.argmax(pred_pt[i, :valid_l], dim=-1).cpu().numpy()
                    )

                    last_t = valid_l - 1
                    true_win.append(float(yR[i].cpu().item()))
                    pred_win_list.append(
                        float(torch.sigmoid(pred_win[i, last_t]).cpu().item())
                    )

    f1_a = f1_score(true_act, pred_act_list, average="macro", zero_division=0)
    f1_p = f1_score(true_pt, pred_pt_list, average="macro", zero_division=0)
    try:
        auc_w = roc_auc_score(true_win, pred_win_list)
    except Exception:
        auc_w = 0.5

    total_score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc_w
    return f1_a, f1_p, auc_w, total_score


def train_model(model, train_loader, val_loader, epochs,
                device, act_weights, pt_weights, fold_id):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    criterion_ce_act = MaskedWeightedCE(weight=act_weights, label_smoothing=0.03)
    criterion_ce_pt = MaskedWeightedCE(weight=pt_weights, label_smoothing=0.03)
    criterion_win = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_path = f"best_shuttlenet_fold{fold_id}.pth"
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, yA, yP, yR, L in train_loader:
            X = X.to(device)
            yA = yA.to(device)
            yP = yP.to(device)
            yR = yR.to(device)
            L = L.to(device)

            optimizer.zero_grad()

            pred_act, pred_pt, pred_win = model(X, L)

            B, seq_len = yA.shape
            L_y = torch.clamp(L - 1, min=1)
            mask_y = torch.arange(seq_len, device=device)[None, :] < L_y[:, None]

            loss_act = criterion_ce_act(pred_act, yA, mask_y)
            loss_pt = criterion_ce_pt(pred_pt, yP, mask_y)

            last_t_idx = torch.clamp(L_y - 1, min=0)
            valid_pred_win = pred_win.gather(1, last_t_idx.unsqueeze(1)).squeeze(1)
            loss_win = criterion_win(valid_pred_win, yR)

            loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_win

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        f1_a, f1_p, auc_w, total_score = evaluate_model(
            model, val_loader, device
        )

        print(
            f"[Fold {fold_id}] Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {train_loss/len(train_loader):.4f} | "
            f"Val F1_A: {f1_a:.4f} | F1_P: {f1_p:.4f} | "
            f"AUC: {auc_w:.4f} | Score: {total_score:.4f}"
        )

        if total_score > best_score + 1e-4:
            best_score = total_score
            torch.save(model.state_dict(), best_path)
            print(f"  => Fold {fold_id} best model updated.")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  => Fold {fold_id} early stop at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(best_path))
    return model, best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_final.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--base_submit", type=str, default=None,
                    help="以前 leaderboard 表現較好的 submission，用於 ensemble")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_train = pd.read_csv(args.train)
    if "matchnumberGame" not in df_train.columns:
        if "match" in df_train.columns and "numberGame" in df_train.columns:
            df_train["matchnumberGame"] = (
                df_train["match"].astype(str) + "_" + df_train["numberGame"].astype(str)
            )
        else:
            raise ValueError(
                "train.csv 需要 matchnumberGame 或 (match, numberGame) 來做分組"
            )

    df_train = preprocess_dataframe(df_train)

    df_test = pd.read_csv(args.test)
    df_test = preprocess_dataframe(df_test)

    df_all = pd.concat([df_train, df_test], axis=0)
    vocab_sizes = {
        feat: int(df_all[feat].max()) + 1
        for feat in ALL_FEATURES
    }
    act_classes_num = int(df_train["actionId"].max()) + 1
    pt_classes_num = int(df_train["pointId"].max()) + 1

    act_counts = np.bincount(
        df_train["actionId"].values, minlength=act_classes_num
    )
    pt_counts = np.bincount(
        df_train["pointId"].values, minlength=pt_classes_num
    )
    act_counts = np.maximum(act_counts, 1)
    pt_counts = np.maximum(pt_counts, 1)

    act_weights_np = np.sqrt(np.sum(act_counts) / act_counts)
    pt_weights_np = np.sqrt(np.sum(pt_counts) / pt_counts)

    act_weights = torch.FloatTensor(
        act_weights_np / np.sum(act_weights_np) * act_classes_num
    ).to(device)
    pt_weights = torch.FloatTensor(
        pt_weights_np / np.sum(pt_weights_np) * pt_classes_num
    ).to(device)

    rally_meta = (
        df_train.groupby("rally_uid")[["matchnumberGame"]]
        .first()
        .reset_index()
    )
    rally_indices = np.arange(len(rally_meta))
    rally_groups = rally_meta["matchnumberGame"].values

    gkf = GroupKFold(n_splits=NUM_FOLDS)

    fold_paths = []

    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(rally_indices, groups=rally_groups), start=1):
        if fold_id > NUM_FOLDS:
            break

        train_uids = set(rally_meta["rally_uid"].values[tr_idx])
        val_uids = set(rally_meta["rally_uid"].values[va_idx])

        df_tr = df_train[df_train["rally_uid"].isin(train_uids)].copy()
        df_va = df_train[df_train["rally_uid"].isin(val_uids)].copy()

        X_tr, yA_tr, yP_tr, yR_tr, L_tr, _ = build_sequences(df_tr, is_train=True)
        X_va, yA_va, yP_va, yR_va, L_va, _ = build_sequences(df_va, is_train=True)

        train_loader = DataLoader(
            RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr),
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            RallyDataset(X_va, yA_va, yP_va, yR_va, L_va),
            batch_size=64,
            shuffle=False,
        )

        model = ShuttleNetFusion(
            vocab_sizes, act_classes_num, pt_classes_num
        ).to(device)

        print(f"==== Start training fold {fold_id}/{NUM_FOLDS} ====")
        model, best_path = train_model(
            model, train_loader, val_loader,
            args.epochs, device, act_weights, pt_weights, fold_id
        )
        fold_paths.append(best_path)

    print("Building test sequences...")
    X_test, L_test, uids_test = build_sequences(df_test, is_train=False)
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(L_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Ensembling folds on test set...")
    all_pred_act = None
    all_pred_pt = None
    all_pred_win = None

    with torch.no_grad():
        for fold_id, path in enumerate(fold_paths, start=1):
            model = ShuttleNetFusion(
                vocab_sizes, act_classes_num, pt_classes_num
            ).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()

            fold_act_list = []
            fold_pt_list = []
            fold_win_list = []

            for X_b, L_b in test_loader:
                X_b = X_b.to(device)
                L_b = L_b.to(device)
                pred_act, pred_pt, pred_win = model(X_b, L_b)
                fold_act_list.append(pred_act.cpu())
                fold_pt_list.append(pred_pt.cpu())
                fold_win_list.append(pred_win.cpu())

            fold_act = torch.cat(fold_act_list, dim=0)
            fold_pt = torch.cat(fold_pt_list, dim=0)
            fold_win = torch.cat(fold_win_list, dim=0)

            if all_pred_act is None:
                all_pred_act = fold_act
                all_pred_pt = fold_pt
                all_pred_win = fold_win
            else:
                all_pred_act += fold_act
                all_pred_pt += fold_pt
                all_pred_win += fold_win

    all_pred_act /= len(fold_paths)
    all_pred_pt /= len(fold_paths)
    all_pred_win /= len(fold_paths)

    pred_rows = []
    uid_idx = 0
    for i in range(X_test.shape[0]):
        l = int(L_test[i])
        last_t = max(0, l - 1)
        a_idx = int(torch.argmax(all_pred_act[i, last_t]).item())
        p_idx = int(torch.argmax(all_pred_pt[i, last_t]).item())
        s_prob = float(torch.sigmoid(all_pred_win[i, last_t]).item())
        pred_rows.append(
            {
                "rally_uid": int(uids_test[uid_idx]),
                "actionId": a_idx,
                "pointId": p_idx,
                "serverGetPoint": s_prob,
            }
        )
        uid_idx += 1

    ensemble_df = pd.DataFrame(pred_rows).sort_values("rally_uid")

    # 如果有舊 submission，就做離線 ensemble
    if args.base_submit is not None:
        base_df = pd.read_csv(args.base_submit)
        base_df = base_df.sort_values("rally_uid")
        merged = ensemble_df.merge(
            base_df,
            on="rally_uid",
            suffixes=("_new", "_base"),
            how="inner",
        )

        def vote(col_new, col_base):
            # 如果一樣就用那個，不一樣用 new
            return np.where(col_new == col_base, col_new, col_new)

        action_final = vote(
            merged["actionId_new"].values,
            merged["actionId_base"].values,
        )
        point_final = vote(
            merged["pointId_new"].values,
            merged["pointId_base"].values,
        )
        server_prob = (
            merged["serverGetPoint_new"].values +
            merged["serverGetPoint_base"].values
        ) / 2.0

        final_df = pd.DataFrame(
            {
                "rally_uid": merged["rally_uid"].values.astype(int),
                "actionId": action_final.astype(int),
                "pointId": point_final.astype(int),
                "serverGetPoint": server_prob.astype(float),
            }
        ).sort_values("rally_uid")
    else:
        final_df = ensemble_df[["rally_uid", "actionId", "pointId", "serverGetPoint"]]

    final_df.to_csv(args.out, index=False)
    print(f"Submission saved to: {args.out}")


if __name__ == "__main__":
    main()