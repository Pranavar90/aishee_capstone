import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class ScreamSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma='scale'):
        self.C = C
        self.gamma = gamma
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True))
        ])

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
        
        # Gated Graph Neural Network
        # GatedGraphConv(out_channels, num_layers)
        # It internally uses a GRU.
        # Note: num_layers in GatedGraphConv refers to the number of message passing steps (aggregator repetitions).
        # But commonly we might want multiple GNN layers?
        # The user said: "Implement GatedGraphConv. The network should pass messages across frames..."
        # GatedGraphConv signature: (out_channels, num_layers)
        
        self.hidden_channels = hidden_channels
        
        # Initial projection to hidden_channels if needed?
        # GatedGraphConv expects input features matching out_channels if we rely on its internal GRU fully?
        # No, PyG's GatedGraphConv takes x (size *), edge_index.
        # "The operator 'Gated Graph Sequence Neural Networks' ...
        # updatet = GRU(agg, ht-1)
        # It requires the input x to be of shape (N, out_channels) initially?
        # Let's check: PyG GatedGraphConv docs say `forward(x, edge_index)`.
        # Usually we project inputs to hidden size first.
        
        self.lin0 = torch.nn.Linear(num_node_features, hidden_channels)
        self.ggnn = GatedGraphConv(hidden_channels, num_layers=num_layers)
        
        # Classification head
        # We need to pool the node representations to graph representation?
        # Or classify per node?
        # The task is audio classification (Scream vs Non-Scream at clip level).
        # So we use Global Pooling.
        # "The network should pass messages across frames to capture the 'shriek' pattern..."
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Input projection
        x = self.lin0(x)
        x = F.relu(x)
        
        # Gated Graph Conv
        # Returns (N, hidden_channels)
        x = self.ggnn(x, edge_index)
        
        # Readout / Global Pooling
        # Use Global Attention or Mean Pooling
        # Let's use Global Mean Pooling for simplicity or GlobalAttention as it's powerful.
        # User didn't specify, but "Human Scream" is a distinct event.
        # Let's use simple global_mean_pool equivalent manually or via scatter 
        # But since we batch, `batch` vector is needed.
        
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        
        # MLP for classification
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
