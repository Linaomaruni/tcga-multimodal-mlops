"""
Late-Fusion MLP for Multi-modal Cancer Classification.
Combines TITAN visual embeddings (768-dim) with text embeddings (896-dim).
"""

import torch
import torch.nn as nn


class LateFusionMLP(nn.Module):
    """
    Late-fusion architecture that processes each modality separately
    before combining them for classification.
    
    Architecture:
        Visual branch: 768 -> hidden_dim
        Text branch: 896 -> hidden_dim
        Fusion: concat -> MLP -> num_classes
    """
    
    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 896,
        hidden_dim: int = 256,
        num_classes: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Visual branch
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Text branch
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Fusion layers (after concatenation: hidden_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, visual_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visual_emb: TITAN embeddings [batch_size, 768]
            text_emb: Text embeddings [batch_size, 896]
            
        Returns:
            logits: Class predictions [batch_size, num_classes]
        """
        # Process each modality
        visual_features = self.visual_encoder(visual_emb)
        text_features = self.text_encoder(text_emb)
        
        # Late fusion: concatenate features
        fused = torch.cat([visual_features, text_features], dim=1)
        
        # Classification
        fused_features = self.fusion(fused)
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_embeddings(self, visual_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Get fused embeddings before classification (useful for visualization).
        
        Returns:
            Fused feature embeddings [batch_size, hidden_dim // 2]
        """
        visual_features = self.visual_encoder(visual_emb)
        text_features = self.text_encoder(text_emb)
        fused = torch.cat([visual_features, text_features], dim=1)
        fused_features = self.fusion(fused)
        return fused_features


class UnimodalMLP(nn.Module):
    """
    Unimodal baseline for comparison.
    Can use either visual or text modality alone.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    # Quick test
    batch_size = 32
    
    # Test LateFusionMLP
    model = LateFusionMLP()
    visual = torch.randn(batch_size, 768)
    text = torch.randn(batch_size, 896)
    output = model(visual, text)
    print(f"LateFusionMLP output shape: {output.shape}")  # [32, 32]
    
    # Test UnimodalMLP
    unimodal = UnimodalMLP(input_dim=768)
    output_uni = unimodal(visual)
    print(f"UnimodalMLP output shape: {output_uni.shape}")  # [32, 32]
    
    print(" Models working correctly!")