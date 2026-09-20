import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 設定圖表風格與中文字體（避免亂碼）
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取資料
print("正在讀取資料...")
df = pd.read_csv(os.getenv("TT_TRAIN_CSV", "data/train.csv"))

# 2. 基本資訊概覽
print("\n--- 資料集基本資訊 ---")
print(f"總資料筆數: {len(df)}")
print(f"獨立回合數 (rally_uid): {df['rally_uid'].nunique()}")
print(f"缺失值統計:\n{df.isnull().sum()[df.isnull().sum() > 0]}") # 檢查是否有空值

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3. 拍數 (Rally Length) 分佈
# 計算每個 rally_uid 的最大 strikeNumber
rally_lengths = df.groupby('rally_uid')['strikeNumber'].max()
sns.histplot(rally_lengths, bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('每回合總拍數 (Rally Length) 分佈')
axes[0, 0].set_xlabel('總拍數')
axes[0, 0].set_ylabel('回合數量')

# 4. 目標變數：勝負比例 (serverGetPoint)
# 每個 rally_uid 的勝負結果是固定的，所以我們先去重
server_win = df.drop_duplicates(subset=['rally_uid'])['serverGetPoint']
sns.countplot(x=server_win, ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('發球方是否得分 (0: 否, 1: 是)')
axes[0, 1].set_xlabel('serverGetPoint')
axes[0, 1].set_ylabel('回合數量')

# 5. 目標變數：球種 (actionId) 分佈
sns.countplot(x='actionId', data=df, order=df['actionId'].value_counts().index, ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('球種 (actionId) 出現次數分佈')
axes[1, 0].set_xlabel('球種 ID')
axes[1, 0].set_ylabel('擊球次數')

# 6. 目標變數：落點 (pointId) 分佈
sns.countplot(x='pointId', data=df, order=df['pointId'].value_counts().index, ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('落點位置 (pointId) 出現次數分佈')
axes[1, 1].set_xlabel('落點 ID')
axes[1, 1].set_ylabel('擊球次數')

plt.tight_layout()
plt.savefig('eda_basic_distributions.png')
print("已將基本分佈圖表儲存為 'eda_basic_distributions.png'")

# 7. 進階：球種狀態轉移矩陣 (Transition Matrix)
# 建立前一拍的 actionId 特徵
df['prev_actionId'] = df.groupby('rally_uid')['actionId'].shift(1)

# 計算轉移矩陣
transition_counts = pd.crosstab(df['prev_actionId'], df['actionId'])
# 轉為機率百分比
transition_prob = transition_counts.div(transition_counts.sum(axis=1), axis=0)

plt.figure(figsize=(12, 10))
sns.heatmap(transition_prob, cmap='Blues', annot=False)
plt.title('球種轉移機率矩陣 (前一拍 -> 當前拍)')
plt.xlabel('當前拍球種 (actionId)')
plt.ylabel('前一拍球種 (prev_actionId)')
plt.savefig('eda_transition_matrix.png')
print("已將球種轉移矩陣儲存為 'eda_transition_matrix.png'")