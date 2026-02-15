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
        batch = data.batch if hasattr(data, 'batch') else None
        
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
