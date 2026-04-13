"""
src/models.py — Model definitions.

MultiHeadGGNN  — production multi-task model (scream + emotion + DANN domain)
ScreamGGNN     — legacy single-task binary detector (kept for compatibility)
ScreamSVM      — sklearn SVM wrapper
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── SVM ───────────────────────────────────────────────────────────────────────

class ScreamSVM:
    """Binary scream detector: StandardScaler + RBF SVC."""

    def __init__(self, C=10.0, gamma="scale"):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(C=C, kernel="rbf", gamma=gamma, probability=True)),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


# ── Single-task GGNN (legacy) ─────────────────────────────────────────────────

class ScreamGGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, num_layers=4):
        super().__init__()
        self.lin0 = Linear(num_node_features, hidden_channels)
        self.bn0  = BatchNorm1d(hidden_channels)
        self.conv = GatedGraphConv(hidden_channels, num_layers)
        self.bn1  = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.bn2  = BatchNorm1d(hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn0(self.lin0(x)))
        x = self.conv(x, edge_index)
        x = F.relu(self.bn1(x))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.lin1(x)))
        return F.log_softmax(self.lin2(x), dim=1)


# ── Multi-task GGNN (production) ──────────────────────────────────────────────

class MultiHeadGGNN(torch.nn.Module):
    """
    Multi-task GNN for joint scream detection and emotion classification.

    Architecture
    ────────────
    Input: temporal frame graph  (T nodes, F features each)
      ├─ Linear(F → H) + BN + ReLU                       [node projection]
      ├─ GatedGraphConv(H, num_layers) + BN + ReLU        [message passing]
      ├─ global_mean_pool ⊕ global_max_pool  → (2H,)      [dual readout]
      └─ Trunk: Linear(2H→H) + BN + ReLU + Dropout(0.4)
               Linear(H→H//2) + BN + ReLU                [shared repr]
         ├─ Head A – scream:  Linear→ReLU→Drop→Linear(2) → log_softmax
         ├─ Head B – emotion: Linear→ReLU→Drop→Linear(K) → log_softmax
         └─ Head C – domain:  Linear→ReLU→Drop→Linear(2) → log_softmax  [DANN]

    Parameters
    ──────────
    lstm_hidden  kept for config backwards-compat; repurposed as trunk width.
    """

    def __init__(
        self,
        num_node_features,
        hidden_channels = 256,
        num_layers      = 8,
        num_emotion_classes = 5,
        lstm_hidden     = 256,   # reused as trunk hidden dim
        use_dann        = True,
    ):
        super().__init__()
        self.use_dann = use_dann
        H = hidden_channels
        T = lstm_hidden  # trunk width

        # ── Input projection ──────────────────────────────────────────────────
        self.lin0 = Linear(num_node_features, H)
        self.bn0  = BatchNorm1d(H)

        # ── Graph convolution ─────────────────────────────────────────────────
        self.conv = GatedGraphConv(H, num_layers)
        self.bn1  = BatchNorm1d(H)

        # ── Shared trunk (input = 2H from dual pool) ──────────────────────────
        self.trunk = torch.nn.Sequential(
            Linear(2 * H, H),
            BatchNorm1d(H),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),  # Increased for better regularization
            Linear(H, T),
            BatchNorm1d(T),
            torch.nn.ReLU(),
        )

        # ── Task heads ────────────────────────────────────────────────────────
        def _head(out_dim):
            return torch.nn.Sequential(
                Linear(T, T // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.4),  # Increased for better regularization
                Linear(T // 2, out_dim),
            )

        self.head_scream  = _head(2)
        self.head_emotion = _head(num_emotion_classes)
        if use_dann:
            self.domain_disc = _head(2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding
        x = F.relu(self.bn0(self.lin0(x)))
        x = F.relu(self.bn1(self.conv(x, edge_index)))

        # Dual graph readout
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)  # (B, 2H)

        shared = self.trunk(x)  # (B, T)

        s_out = F.log_softmax(self.head_scream(shared),  dim=1)
        e_out = F.log_softmax(self.head_emotion(shared), dim=1)

        if self.use_dann:
            d_out = F.log_softmax(self.domain_disc(shared), dim=1)
            return s_out, e_out, d_out

        return s_out, e_out

    # ── Convenience helpers ───────────────────────────────────────────────────
    def get_scream_prob(self, data):
        out = self.forward(data)
        return torch.exp(out[0])[0][1].item()

    def get_emotion_prob(self, data):
        out = self.forward(data)
        if not isinstance(out, (list, tuple)) or len(out) < 2:
            return None
        return torch.exp(out[1])[0].cpu().numpy()
