import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single 1D convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class ECGTransformerEncoder(nn.Module):
    """Lightweight Transformer encoder on top of CNN features."""

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class ECGGuard(nn.Module):
    """
    ECG-Guard: 1D-CNN + Transformer hybrid for cardiac anomaly detection.

    Uses Monte Carlo Dropout for uncertainty quantification.
    Refuses prediction when confidence falls below clinical threshold.
    """

    def __init__(self, num_classes=5, dropout_rate=0.3,
                 confidence_threshold=0.75):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.dropout_rate = dropout_rate

        # CNN feature extractor
        self.cnn = nn.Sequential(
            ConvBlock(12, 32, kernel_size=7),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3),
        )

        # Transformer encoder
        self.transformer = ECGTransformerEncoder(d_model=128)

        # Dropout (kept active during inference for MC Dropout)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 12 leads, time_steps)
        features = self.cnn(x)
        # Reshape for transformer: (batch, seq_len, d_model)
        features = features.permute(0, 2, 1)
        features = self.transformer(features)
        # Global average pooling
        features = features.mean(dim=1)
        features = self.dropout(features)
        return self.classifier(features)

    def predict_with_uncertainty(self, x, n_passes=30):
        """
        Monte Carlo Dropout inference.
        Runs n_passes forward passes with dropout active.
        Returns prediction and confidence score.
        Refuses to predict if confidence < threshold.
        """
        self.train()  # keep dropout active
        with torch.no_grad():
            preds = torch.stack([
                torch.softmax(self.forward(x), dim=-1)
                for _ in range(n_passes)
            ])

        mean_pred = preds.mean(dim=0)
        confidence = mean_pred.max(dim=-1).values

        predicted_class = mean_pred.argmax(dim=-1)

        # Refusal mechanism — core alignment feature
        refused = confidence < self.confidence_threshold
        predicted_class[refused] = -1  # -1 means "refused"

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "refused": refused,
            "uncertainty": preds.std(dim=0)
        }


CONDITION_LABELS = {
    0: "Normal Sinus Rhythm",
    1: "Atrial Fibrillation",
    2: "ST-Elevation MI",
    3: "Left Bundle Branch Block",
    4: "Bradycardia / Tachycardia",
    -1: "REFUSED — confidence below clinical threshold"
}
