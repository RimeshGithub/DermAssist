"""Model architecture - copied verbatim from Section 8 of the notebook.

Do NOT change layer names, sizes or order: the saved state_dict must load with strict key matching.
"""
import timm
import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, m):
        return self.net(m)


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, metadata_dim, meta_embed_dim=64,
                 backbone="swin_tiny_patch4_window7_224", pretrained=True,
                 dropout=0.3, drop_path_rate=0.0):
        super().__init__()
        # Image branch: timm backbone with the classifier head removed (num_classes=0)
        self.image_encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        img_embed_dim = self.image_encoder.num_features

        # Metadata branch
        self.metadata_encoder = MetadataEncoder(
            metadata_dim, hidden_dim=64, out_dim=meta_embed_dim, dropout=dropout)

        # Fusion + classification head
        fused_dim = img_embed_dim + meta_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.img_embed_dim = img_embed_dim
        self.meta_embed_dim = meta_embed_dim

    def forward(self, image, metadata):
        z_img = self.image_encoder(image)
        z_meta = self.metadata_encoder(metadata)
        z = torch.cat([z_img, z_meta], dim=1)
        return self.classifier(z)
