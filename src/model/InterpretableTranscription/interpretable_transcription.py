import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # [B, C, F, T]

class TranscriptionModel(nn.Module):
    def __init__(self, n_mels=128, cnn_channels=64, lstm_hidden=256, num_layers=2):
        super().__init__()
        self.cnn = ConvBlock(in_channels=1, out_channels=cnn_channels)

        self.lstm = nn.LSTM(
            input_size=cnn_channels * n_mels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        lstm_out = lstm_hidden * 2  # bidirectional

        # Output heads
        self.onset_head = nn.Linear(lstm_out, 128)
        self.frame_head = nn.Linear(lstm_out, 128)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.Sigmoid()  # velocity normalized [0,1]
        )

    def forward(self, x):
        """
        x: [B, 128, T] → mel spec
        """
        x = x.unsqueeze(1)  # [B, 1, 128, T]
        x = self.cnn(x)     # [B, C, 128, T]
        x = x.permute(0, 3, 1, 2)  # [B, T, C, 128]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T, C*n_mels]

        x, _ = self.lstm(x)  # [B, T, H*2]

        return {
            "onset": torch.sigmoid(self.onset_head(x)),     # [B, T, 128]
            "frame": torch.sigmoid(self.frame_head(x)),     # [B, T, 128]
            "velocity": self.velocity_head(x)               # [B, T, 128]
        }
