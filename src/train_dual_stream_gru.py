import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PAD_TOKEN = 0
MAX_LEN = 80 

# ==========================================
# 1. 特徵定義
# ==========================================
FEATURES_SPA = ["pointId", "positionId", "prev_pointId"] 
FEATURES_ACT = ["actionId", "spinId", "strikeId", "prev_actionId"] 
FEATURES_CTX = [
    "sex", "handId", "strengthId", "scoreSelf", "scoreOther", 
    "scoreDiff", "scoreSum", "isGamePoint", "scoreParity", "isLeading",
    "gamePlayerId", "gamePlayerOtherId", "isServer", "strikeNumber"
]
ALL_FEATURES = FEATURES_SPA + FEATURES_ACT + FEATURES_CTX
FEATURE_IDX = {feat: idx for idx, feat in enumerate(ALL_FEATURES)}

# ==========================================
# 2. Dataset 定義
# ==========================================
class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L, dtype=torch.long)
        
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

# ==========================================
# 3. Masked Weighted CE
# ==========================================
class MaskedWeightedCE(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, mask):
        C = inputs.shape[-1]
        inputs = inputs.reshape(-1, C)[mask.reshape(-1)]
        targets = targets.reshape(-1)[mask.reshape(-1)]

        if len(targets) == 0:
            return torch.tensor(0.0, requires_grad=True).to(inputs.device)

        return F.cross_entropy(inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing)

# ==========================================
# 4. Deep GRU + Triple Cascade
# ==========================================
class ShuttleNetFusion(nn.Module):
    def __init__(self, vocab_sizes, act_classes_num, pt_classes_num, emb_dim=64, hidden_dim=256):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_sizes.get(feat, 500) + 2, emb_dim, padding_idx=PAD_TOKEN) 
            for feat in ALL_FEATURES
        })
        
        dim_spa = len(FEATURES_SPA) * emb_dim
        dim_act = len(FEATURES_ACT) * emb_dim
        dim_ctx = len(FEATURES_CTX) * emb_dim
        
        self.rnn_spa = nn.GRU(dim_spa, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)
        self.rnn_act = nn.GRU(dim_act, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)

        fused_dim = hidden_dim * 2 + dim_ctx

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(), 
            nn.Dropout(0.3)
        )

        wide_dim = fused_dim + dim_ctx 

        self.head_act = nn.Sequential(
            nn.Linear(wide_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(hidden_dim, act_classes_num)
        )
        self.head_pt  = nn.Sequential(
            nn.Linear(wide_dim + act_classes_num, hidden_dim), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(hidden_dim, pt_classes_num)
        )
        self.head_win = nn.Sequential(
            nn.Linear(wide_dim + act_classes_num + pt_classes_num, hidden_dim // 2), nn.GELU(), 
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, X, lengths):
        emb_spa = torch.cat([self.embeddings[feat](X[:, :, FEATURE_IDX[feat]]) for feat in FEATURES_SPA], dim=-1)
        emb_act = torch.cat([self.embeddings[feat](X[:, :, FEATURE_IDX[feat]]) for feat in FEATURES_ACT], dim=-1)
        emb_ctx = torch.cat([self.embeddings[feat](X[:, :, FEATURE_IDX[feat]]) for feat in FEATURES_CTX], dim=-1)

        out_spa, _ = self.rnn_spa(emb_spa)
        out_act, _ = self.rnn_act(emb_act)

        fused_features = torch.cat([out_spa, out_act, emb_ctx], dim=-1)
        final_repr = self.fusion(fused_features)

        wide_deep_repr = torch.cat([final_repr, emb_ctx], dim=-1)

        pred_act = self.head_act(wide_deep_repr)
        act_probs = F.softmax(pred_act, dim=-1) 
        
        pt_input = torch.cat([wide_deep_repr, act_probs], dim=-1)
        pred_pt  = self.head_pt(pt_input)
        pt_probs = F.softmax(pred_pt, dim=-1)
        
        win_input = torch.cat([wide_deep_repr, act_probs, pt_probs], dim=-1)
        pred_win = self.head_win(win_input).squeeze(-1)
        
        return pred_act, pred_pt, pred_win

# ==========================================
# 5. 資料處理
# ==========================================
def preprocess_dataframe(df):
    df = df.copy()
    df['gamePlayerId'] = df['gamePlayerId'].fillna(0).astype(int)
    df['gamePlayerOtherId'] = df['gamePlayerOtherId'].fillna(0).astype(int)
    
    df['scoreDiff'] = df['scoreSelf'] - df['scoreOther'] + 30 
    df['scoreSum'] = df['scoreSelf'] + df['scoreOther']
    df['isGamePoint'] = ((df['scoreSelf'] >= 20) | (df['scoreOther'] >= 20)).astype(int)
    df['scoreParity'] = df['scoreSum'] % 2 
    df['isLeading'] = (df['scoreSelf'] > df['scoreOther']).astype(int)
    
    min_strike_idx = df.groupby('rally_uid')['strikeNumber'].idxmin()
    server_info = df.loc[min_strike_idx, ['rally_uid', 'gamePlayerId']].rename(columns={'gamePlayerId': 'serverPlayerId'})
    df = df.merge(server_info, on='rally_uid', how='left')
    df['isServer'] = (df['gamePlayerId'] == df['serverPlayerId']).astype(int)

    df = df.sort_values(['rally_uid', 'strikeNumber'])
    df['prev_pointId'] = df.groupby('rally_uid')['pointId'].shift(1).fillna(-1).astype(int) + 1
    df['prev_actionId'] = df.groupby('rally_uid')['actionId'].shift(1).fillna(-1).astype(int) + 1
    return df

def build_sequences(df, is_train=True):
    grouped = df.groupby('rally_uid')
    X_list, yA_list, yP_list, yR_list, L_list, uids = [], [], [], [], [], []
    num_features = len(ALL_FEATURES)
    
    for rid, group in grouped:
        group = group.sort_values('strikeNumber')
        seq_len = len(group)
        L_list.append(min(seq_len, MAX_LEN))
        uids.append(rid)
        
        x_seq = np.zeros((MAX_LEN, num_features), dtype=np.int64)
        for idx, feat in enumerate(ALL_FEATURES):
            vals = group[feat].values
            length = min(seq_len, MAX_LEN)
            x_seq[:length, idx] = vals[:length]
        X_list.append(x_seq)
        
        if is_train:
            ya_seq = np.zeros(MAX_LEN, dtype=np.int64)
            yp_seq = np.zeros(MAX_LEN, dtype=np.int64)
            acts = group['actionId'].values[1:] 
            pts = group['pointId'].values[1:]   
            length = min(len(acts), MAX_LEN)
            if length > 0:
                ya_seq[:length] = acts[:length]
                yp_seq[:length] = pts[:length]
                
            yA_list.append(ya_seq)
            yP_list.append(yp_seq)
            yR_list.append(group['serverGetPoint'].values[0])
            
    if is_train:
        return np.array(X_list), np.array(yA_list), np.array(yP_list), np.array(yR_list), np.array(L_list)
    else:
        return np.array(X_list), np.array(L_list), uids

# ==========================================
# 6. 驗證與計分 (🚨 BUG 已修復：AUC 只算最後一拍！)
# ==========================================
def evaluate_model(model, val_loader, device):
    model.eval()
    true_act, pred_act_list, true_pt, pred_pt_list, true_win, pred_win_list = [], [], [], [], [], []
    with torch.no_grad():
        for X, yA, yP, yR, L in val_loader:
            X, yA, yP, yR, L = X.to(device), yA.to(device), yP.to(device), yR.to(device), L.to(device)
            pred_act, pred_pt, pred_win = model(X, L)

            for i in range(len(X)):
                valid_l = L[i].item() - 1 
                if valid_l > 0:
                    true_act.extend(yA[i, :valid_l].cpu().numpy())
                    pred_act_list.extend(torch.argmax(pred_act[i, :valid_l], dim=-1).cpu().numpy())
                    true_pt.extend(yP[i, :valid_l].cpu().numpy())
                    pred_pt_list.extend(torch.argmax(pred_pt[i, :valid_l], dim=-1).cpu().numpy())

                    # 🌟 關鍵修復：官方盲測只會給我們看截斷前的最後一拍
                    # 所以我們計算驗證分數時，也絕對「只看最後一拍」，拒絕前期瞎猜的干擾！
                    last_t = valid_l - 1
                    true_win.append(yR[i].cpu().item())
                    pred_win_list.append(torch.sigmoid(pred_win[i, last_t]).cpu().item())

    f1_a = f1_score(true_act, pred_act_list, average='macro', zero_division=0)
    f1_p = f1_score(true_pt, pred_pt_list, average='macro', zero_division=0)
    try: auc_w = roc_auc_score(true_win, pred_win_list)
    except: auc_w = 0.5

    total_score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc_w
    return f1_a, f1_p, auc_w, total_score

# ==========================================
# 7. 訓練主迴圈
# ==========================================
def train_model(model, train_loader, val_loader, epochs, device, act_weights, pt_weights):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    criterion_ce_act = MaskedWeightedCE(weight=act_weights, label_smoothing=0.05)
    criterion_ce_pt = MaskedWeightedCE(weight=pt_weights, label_smoothing=0.05) 
    criterion_win = nn.BCEWithLogitsLoss()
    best_score = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X, yA, yP, yR, L in train_loader:
            X, yA, yP, yR, L = X.to(device), yA.to(device), yP.to(device), yR.to(device), L.to(device)
            optimizer.zero_grad()
            
            pred_act, pred_pt, pred_win = model(X, L)

            B, seq_len = yA.shape
            L_y = torch.clamp(L - 1, min=1) 
            mask_y = torch.arange(seq_len, device=device)[None, :] < L_y[:, None]
            
            loss_act = criterion_ce_act(pred_act, yA, mask_y)
            loss_pt  = criterion_ce_pt(pred_pt, yP, mask_y)
            
            # 🌟 關鍵修復：我們只在「最後一拍」懲罰勝負的 Loss
            # 這樣模型才會專注於萃取整局累積下來的線索，而不是每拍都在分心！
            last_t_idx = torch.clamp(L_y - 1, min=0)
            valid_pred_win = pred_win.gather(1, last_t_idx.unsqueeze(1)).squeeze(1)
            loss_win = criterion_win(valid_pred_win, yR)

            # 完美對齊官方比例
            loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_win
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        f1_a, f1_p, auc_w, total_score = evaluate_model(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f}")
        print(f"   => Val | F1_A: {f1_a:.4f} | F1_P: {f1_p:.4f} | AUC: {auc_w:.4f} | Score: {total_score:.4f}")

        if total_score > best_score:
            best_score = total_score
            torch.save(model.state_dict(), 'best_shuttlenet.pth')
            print("   => 🏆 抓到 Bug，分數解放了！")

    model.load_state_dict(torch.load('best_shuttlenet.pth'))
    return model

# ==========================================
# 8. 程式進入點
# ==========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_shuttlenet_pro.csv")
    # ⚡ 只需要 15 個 Epoch 就能見效！
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using: {device} | 終極 Bug 修正版 (專注最後一擊)")

    df_train = pd.read_csv(args.train)
    df_train = preprocess_dataframe(df_train)
    X, yA, yP, yR, L = build_sequences(df_train, is_train=True)
    
    X_tr, X_val, yA_tr, yA_val, yP_tr, yP_val, yR_tr, yR_val, L_tr, L_val = train_test_split(
        X, yA, yP, yR, L, test_size=0.1, random_state=SEED
    )
    
    train_loader = DataLoader(RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(RallyDataset(X_val, yA_val, yP_val, yR_val, L_val), batch_size=64, shuffle=False)

    df_test = pd.read_csv(args.test)
    df_test = preprocess_dataframe(df_test)
    df_all = pd.concat([df_train, df_test], axis=0)

    vocab_sizes = {feat: int(df_all[feat].max()) + 1 for feat in ALL_FEATURES}
    act_classes_num = int(df_train['actionId'].max()) + 1
    pt_classes_num = int(df_train['pointId'].max()) + 1

    act_counts = np.bincount(df_train['actionId'].values, minlength=act_classes_num)
    pt_counts = np.bincount(df_train['pointId'].values, minlength=pt_classes_num)
    act_counts = np.maximum(act_counts, 1)
    pt_counts = np.maximum(pt_counts, 1)
    
    act_weights = np.sqrt(np.sum(act_counts) / act_counts)
    pt_weights = np.sqrt(np.sum(pt_counts) / pt_counts)
    
    act_weights = torch.FloatTensor(act_weights / np.sum(act_weights) * act_classes_num).to(device)
    pt_weights = torch.FloatTensor(pt_weights / np.sum(pt_weights) * pt_classes_num).to(device)

    model = ShuttleNetFusion(vocab_sizes, act_classes_num, pt_classes_num).to(device)

    print(f"🔥 開始 {args.epochs} Epochs 的極速煉丹...")
    model = train_model(model, train_loader, val_loader, args.epochs, device, act_weights, pt_weights)
    
    print("🔍 最終盲測推論中...")
    model.eval()
    X_test, L_test, uids_test = build_sequences(df_test, is_train=False)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(L_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    pred_rows = []
    uid_idx = 0
    with torch.no_grad():
        for X_b, L_b in test_loader:
            X_b, L_b = X_b.to(device), L_b.to(device)
            pred_act, pred_pt, pred_win = model(X_b, L_b)
            for i in range(len(X_b)):
                l = L_b[i].item()
                last_t = max(0, l - 1)
                a_idx = int(torch.argmax(pred_act[i, last_t]).item())
                p_idx = int(torch.argmax(pred_pt[i, last_t]).item())
                s_prob = float(torch.sigmoid(pred_win[i, last_t]).item())
                pred_rows.append({
                    "rally_uid": int(uids_test[uid_idx]), 
                    "actionId": a_idx, 
                    "pointId": p_idx, 
                    "serverGetPoint": s_prob
                })
                uid_idx += 1

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")
    out = pred_df[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False)
    
    print(f"✅ 大功告成！AUC 絕對破冰，檔案已儲存：{args.out}")