"""Reimplementations of three published embedded arrhythmia models."""
import torch, torch.nn as nn, numpy as np

# ---------------------------------------------------------------- Busia 2024
# "A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers"
# IEEE TBCAS, DOI 10.1109/TBCAS.2024.3401858 ; arXiv:2402.10748
# Reported: 6,643 params, 49 kB, 98.97% acc on 5 AAMI classes (8-bit),
#           98.36% under electrode-motion noise. GAP9, 4.28 ms, 0.09 mJ.
# Design per paper: window around R peak (Pan-Tompkins), plus pre-RR and
# post-RR intervals as extra features; conv tokenizer -> MHSA encoder -> MLP.
class BusiaTinyTransformer(nn.Module):
    def __init__(self, n_cls=5, d=8, heads=4, win=180):
        super().__init__()
        # convolutional tokenizer: downsample the beat window into tokens
        self.tok = nn.Sequential(
            nn.Conv1d(1, d, kernel_size=7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(d, d, kernel_size=5, stride=3, padding=2), nn.ReLU(),
        )
        self.ntok = win // 12
        self.pos = nn.Parameter(torch.zeros(1, self.ntok, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*2), nn.ReLU(), nn.Linear(d*2, d))
        self.n2 = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + 2, 16), nn.ReLU(), nn.Linear(16, n_cls))

    def forward(self, x, rr):
        t = self.tok(x.unsqueeze(1)).transpose(1, 2)          # B,T,d
        t = t[:, :self.ntok] + self.pos[:, :t.shape[1]]
        a, _ = self.attn(t, t, t); t = self.n1(t + a)
        t = self.n2(t + self.ff(t))
        g = t.mean(1)                                          # global pool
        return self.head(torch.cat([g, rr], 1))


# ---------------------------------------------------------------- Farag 2023
# "A Self-Contained STFT CNN for ECG Classification and Arrhythmia Detection
#  at the Edge", Sensors 23(3):1365, DOI 10.3390/s23031365
# Reported: 1,267-1,619 params, 14-28 kB TFLite, DS1 train / DS2 test,
#           98.18% accuracy, 92.17% mean F1 on DS2.
# Design per paper: STFT front end, then a single small conv layer + dense.
class FaragSTFTCNN(nn.Module):
    def __init__(self, n_cls=5, nfft=32, hop=8, ch=6):
        super().__init__()
        self.nfft, self.hop = nfft, hop
        self.register_buffer('win', torch.hann_window(nfft))
        self.conv = nn.Conv2d(1, ch, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(ch*16, n_cls)

    def forward(self, x, rr=None):
        s = torch.stft(x, self.nfft, self.hop, window=self.win,
                       return_complex=True, center=True)
        m = torch.log1p(s.abs()).unsqueeze(1)                  # B,1,F,T
        h = torch.relu(self.conv(m))
        return self.fc(self.pool(h).flatten(1))


# ---------------------------------------------------------------- ArrythML 2026
# "ArrythML: An Autoencoder-Based TinyML Approach for On-Device Arrhythmia
#  Detection on Resource-Constrained Embedded Systems", arXiv:2606.02256
# Reported: 84% recall, 79% F1, ~180 kB, 9 ms on ESP32-S3, INT8.
# Design: autoencoder trained on NORMAL beats only; large reconstruction
# error => anomaly. Threshold set on a validation split.
class ArrythMLAutoencoder(nn.Module):
    def __init__(self, win=180, lat=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(win, 96), nn.ReLU(),
                                 nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, lat))
        self.dec = nn.Sequential(nn.Linear(lat, 48), nn.ReLU(),
                                 nn.Linear(48, 96), nn.ReLU(), nn.Linear(96, win))
    def forward(self, x, rr=None):
        return self.dec(self.enc(x))
    def score(self, x):
        return ((self.forward(x) - x) ** 2).mean(1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())

if __name__ == '__main__':
    for n, m in [('Busia', BusiaTinyTransformer()),
                 ('Farag', FaragSTFTCNN()),
                 ('ArrythML', ArrythMLAutoencoder())]:
        print(f"{n:<10} {nparams(m):>8,} params   (~{nparams(m)*1/1024:.0f} kB int8)")
