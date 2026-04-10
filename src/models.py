import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class ScreamSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma="scale"):
        self.C = C
        self.gamma = gamma
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True)),
            ]
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


class ScreamGGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers, num_classes=2):
        super(ScreamGGNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.lin0 = torch.nn.Linear(num_node_features, hidden_channels)
        self.bn0 = torch.nn.BatchNorm1d(hidden_channels)
        self.ggnn = GatedGraphConv(hidden_channels, num_layers=num_layers)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        # Classification head
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        # Input projection
        x = self.lin0(x)
        x = self.bn0(x)
        x = F.relu(x)

        # Gated Graph Conv
        x = self.ggnn(x, edge_index)
        x = self.bn1(x)

        # Readout / Global Pooling
        from torch_geometric.nn import global_mean_pool

        x = global_mean_pool(x, batch)

        # MLP for classification
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class MultiHeadGGNN(torch.nn.Module):
    """
    Multi-task GGNN with two heads:
    - Head A: Binary scream detection (scream vs ambient)
    - Head B: Emotion classification (fear, pain, anger, joy, distress)

    Includes optional Domain Adversarial Training (DANN) for domain adaptation.
    """

    def __init__(
        self,
        num_node_features,
        hidden_channels=64,
        num_layers=4,
        num_emotion_classes=6,
        lstm_hidden=64,
        use_dann=True,
    ):
        super(MultiHeadGGNN, self).__init__()

        self.use_dann = use_dann
        self.hidden_channels = hidden_channels

        # Input projection
        self.lin0 = torch.nn.Linear(num_node_features, hidden_channels)
        self.bn0 = torch.nn.BatchNorm1d(hidden_channels)

        # Gated Graph Conv layers
        self.ggnn = GatedGraphConv(hidden_channels, num_layers=num_layers)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        # LSTM for temporal context
        self.lstm = torch.nn.LSTM(
            hidden_channels, lstm_hidden, num_layers=1, batch_first=True
        )

        # Shared representation layer
        self.shared_lin = torch.nn.Linear(lstm_hidden, hidden_channels)
        self.shared_bn = torch.nn.BatchNorm1d(hidden_channels)

        # Head A: Binary Scream Classification
        self.head_scream = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels // 2, 2),  # 2 classes: scream, ambient
        )

        # Head B: Emotion Classification (5 classes: fear, pain, anger, joy, distress)
        self.head_emotion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels // 2, num_emotion_classes),
        )

        # Domain Discriminator (for DANN)
        if use_dann:
            self.domain_discriminator = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(
                    hidden_channels // 2, 2
                ),  # 2 domains: real, emotion datasets
            )

    def forward(self, data, return_features=False):
        """
        Forward pass.

        Args:
            data: PyG Data object with x and edge_index
            return_features: If True, return shared features for analysis

        Returns:
            scream_logits: (batch, 2) - log softmax
            emotion_logits: (batch, num_emotion_classes) - log softmax
            domain_logits: (batch, 2) - log softmax (only if use_dann=True)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        # Input projection
        x = self.lin0(x)
        x = self.bn0(x)
        x = F.relu(x)

        # Gated Graph Conv
        x = self.ggnn(x, edge_index)
        x = self.bn1(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # LSTM for temporal context
        x = x.unsqueeze(1)  # (batch, 1, hidden)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.squeeze(1)  # (batch, lstm_hidden)

        # Shared representation
        shared = self.shared_lin(x)
        shared = self.shared_bn(shared)
        shared = F.relu(shared)

        # Head outputs
        scream_logits = F.log_softmax(self.head_scream(shared), dim=1)
        emotion_logits = F.log_softmax(self.head_emotion(shared), dim=1)

        if self.use_dann:
            domain_logits = F.log_softmax(self.domain_discriminator(shared), dim=1)
            if return_features:
                return scream_logits, emotion_logits, domain_logits, shared
            return scream_logits, emotion_logits, domain_logits

        if return_features:
            return scream_logits, emotion_logits, shared
        return scream_logits, emotion_logits

    def get_scream_prob(self, data):
        """Helper to get scream probability."""
        scream_logits, *_ = self.forward(data)
        return torch.exp(scream_logits)[:, 1]

    def get_emotion_prob(self, data):
        """Helper to get emotion probabilities."""
        _, emotion_logits, *_ = self.forward(data)
        return torch.exp(emotion_logits)


class DomainDiscriminator(torch.nn.Module):
    """
    Standalone domain discriminator for checking domain-invariant features.
    Used during evaluation to verify DANN is working.
    """

    def __init__(self, input_dim=64, hidden_dim=32):
        super(DomainDiscriminator, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)
