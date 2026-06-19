import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')  # GUI 없는 서버 환경
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


# ── 재현성 ──────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

os.makedirs('results', exist_ok=True)

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
CE_EPOCHS       = 200   # CE는 200이 과적합 없는 상한선
SUPCON_EPOCHS   = 700   # 논문 권장 500~1000, 표현 학습은 CE loss 없어서 과적합 거의 없음
PROBE_EPOCHS    = 50    # linear head는 빠르게 수렴, 50이면 충분
BATCH_SIZE      = 512
TEMPERATURE     = 0.07

TEMP_LIST       = [0.05, 0.07, 0.1, 0.2]
BATCH_LIST      = [64, 128, 256, 512]
ABL_EPOCHS      = 200   # ablation도 충분히
ABL_PROBE_EPOCHS= 30


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터셋
# ══════════════════════════════════════════════════════════════════════════════
class TwoCropTransform:
    """같은 이미지에 다른 augmentation을 두 번 적용 (SupCon용)"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def build_dataloaders(batch_size=512):
    supcon_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # PIL→Tensor 먼저: ColorJitter가 tensor 경로를 사용해 worker crash 방지
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ce_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    supcon_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=TwoCropTransform(supcon_transform))
    ce_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False,
        transform=ce_train_transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False,
        transform=test_transform)
    embed_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False,
        transform=test_transform)

    supcon_loader = DataLoader(supcon_trainset, batch_size=batch_size, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=True)
    ce_loader     = DataLoader(ce_trainset, batch_size=batch_size, shuffle=True,
                               num_workers=8, pin_memory=True)
    test_loader   = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)
    embed_loader  = DataLoader(embed_trainset, batch_size=512, shuffle=False, num_workers=4)

    return supcon_loader, ce_loader, test_loader, embed_loader


# ══════════════════════════════════════════════════════════════════════════════
# 2. 모델
# ══════════════════════════════════════════════════════════════════════════════
class ResNet18CIFAR(nn.Module):
    """CIFAR-10용 ResNet-18: 32x32 입력에 맞게 first conv + maxpool 수정"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        self.feat_dim = 512

    def forward(self, x):
        feat = self.encoder(x)
        return feat.view(feat.size(0), -1)  # (B, 512)


class ProjectionHead(nn.Module):
    """SupCon용 MLP projector: 512 -> 128, L2 normalize"""
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18CIFAR()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat


# ══════════════════════════════════════════════════════════════════════════════
# 3. SupConLoss
# ══════════════════════════════════════════════════════════════════════════════
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
    features: (B, n_views, dim), L2 normalized
    labels:   (B,)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B, n_views, dim = features.shape
        device = features.device

        features_flat = features.view(B * n_views, dim)          # (N, dim)
        labels_rep = labels.repeat_interleave(n_views)           # (N,)

        mask_pos = (labels_rep.unsqueeze(1) == labels_rep.unsqueeze(0)).float()
        self_mask = torch.eye(B * n_views, dtype=torch.bool, device=device)
        mask_pos.masked_fill_(self_mask, 0)

        sim = torch.matmul(features_flat, features_flat.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability

        exp_sim = torch.exp(sim).masked_fill(self_mask, 0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        n_pos = mask_pos.sum(dim=1)
        valid = n_pos > 0
        loss = -(mask_pos * log_prob).sum(dim=1)[valid] / n_pos[valid]
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Early Stopping
# ══════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    mode='max': accuracy처럼 높을수록 좋은 지표
    mode='min': loss처럼 낮을수록 좋은 지표
    best 갱신 시 체크포인트 저장, patience 초과 시 stop 신호 반환
    """
    def __init__(self, patience, mode='max', delta=1e-4, save_path=None):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.save_path = save_path
        self.best      = -np.inf if mode == 'max' else np.inf
        self.counter   = 0
        self.best_state = None

    def step(self, metric, *models):
        improved = (self.mode == 'max' and metric > self.best + self.delta) or \
                   (self.mode == 'min' and metric < self.best - self.delta)

        if improved:
            self.best    = metric
            self.counter = 0
            self.best_state = [
                {k: v.cpu().clone() for k, v in m.state_dict().items()}
                for m in models
            ]
            if self.save_path:
                for i, m in enumerate(models):
                    path = self.save_path if len(models) == 1 \
                           else self.save_path.replace('.pth', f'_{i}.pth')
                    torch.save(m.state_dict(), path)
        else:
            self.counter += 1

        return self.counter >= self.patience  # True = stop

    def restore(self, *models):
        """best weight 복원"""
        if self.best_state is None:
            return
        for m, state in zip(models, self.best_state):
            m.load_state_dict({k: v.to(next(m.parameters()).device)
                               for k, v in state.items()})


# ══════════════════════════════════════════════════════════════════════════════
# 5. 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_linear(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = classifier(encoder(imgs))
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, lbls = [], []
    for imgs, labels in loader:
        feats.append(encoder(imgs.to(device)).cpu().numpy())
        lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


def plot_tsne(feats, labels, title, save_path, n_samples=5000):
    idx = np.random.choice(len(feats), min(n_samples, len(feats)), replace=False)
    print(f't-SNE 계산 중: {title} ...')
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    emb = tsne.fit_transform(feats[idx])
    lbl = labels[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = cm.get_cmap('tab10', 10)
    for c in range(10):
        m = lbl == c
        ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(c)],
                   label=CIFAR10_CLASSES[c], s=6, alpha=0.7)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. 학습 함수
# ══════════════════════════════════════════════════════════════════════════════
def train_ce(ce_loader, test_loader, device):
    print('\n' + '='*55)
    print('  Cross-Entropy Baseline 학습')
    print('='*55)
    set_seed(42)

    model = CEModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CE_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=30, mode='max', save_path='results/ce_model.pth')

    train_losses = []
    for epoch in range(1, CE_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in ce_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(ce_loader))

        acc = evaluate(model, test_loader, device)
        stop = es.step(acc, model)

        if epoch % 10 == 0:
            print(f'[CE] Epoch {epoch:3d}/{CE_EPOCHS} | Loss: {train_losses[-1]:.4f} | '
                  f'Test Acc: {acc:.2f}% | Best: {es.best:.2f}% | Patience: {es.counter}/{es.patience}')

        if stop:
            print(f'[CE] Early stopping at epoch {epoch}. Best acc: {es.best:.2f}%')
            break

    es.restore(model)
    final_acc = evaluate(model, test_loader, device)
    print(f'\n[CE] Final Test Accuracy (best weights): {final_acc:.2f}%')
    return model, final_acc, train_losses


def train_supcon(supcon_loader, ce_loader, test_loader, device,
                 temperature=0.07, batch_size=512, tag=''):
    label = f'SupCon{tag}'
    print(f'\n{"="*55}')
    print(f'  {label} Stage1: Representation Learning  (τ={temperature}, bs={batch_size})')
    print('='*55)
    set_seed(42)

    encoder   = ResNet18CIFAR().to(device)
    projector = ProjectionHead().to(device)
    criterion = SupConLoss(temperature=temperature)
    params    = list(encoder.parameters()) + list(projector.parameters())
    # lr = 0.05 * batch_size/256 (linear scaling rule, SGD 기준)
    lr = 0.05 * (BATCH_SIZE / 256)
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)

    es_s1 = EarlyStopping(patience=150, mode='min', delta=1e-6)

    stage1_losses = []
    for epoch in range(1, SUPCON_EPOCHS + 1):
        encoder.train(); projector.train()
        epoch_loss = 0.0
        for (img1, img2), labels in supcon_loader:
            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            B = labels.size(0)
            z1 = projector(encoder(img1))  # (B, 128)
            z2 = projector(encoder(img2))  # (B, 128)
            z = torch.stack([z1, z2], dim=1)  # (B, 2, 128) - 올바른 페어링
            loss = criterion(z, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(supcon_loader)
        stage1_losses.append(avg_loss)
        stop = es_s1.step(avg_loss, encoder, projector)

        if epoch % 10 == 0:
            print(f'[{label} S1] Epoch {epoch:3d}/{SUPCON_EPOCHS} | Loss: {avg_loss:.4f} | '
                  f'Best: {es_s1.best:.4f} | Patience: {es_s1.counter}/{es_s1.patience}')

        if stop:
            print(f'[{label} S1] Early stopping at epoch {epoch}. Best loss: {es_s1.best:.4f}')
            break

    es_s1.restore(encoder, projector)
    torch.save(encoder.state_dict(),   f'results/supcon_encoder{tag}.pth')
    torch.save(projector.state_dict(), f'results/supcon_projector{tag}.pth')

    # Stage 2: Linear Probe
    print(f'\n  {label} Stage2: Linear Probe')
    for p in encoder.parameters():
        p.requires_grad = False

    classifier  = LinearClassifier().to(device)
    probe_opt   = optim.Adam(classifier.parameters(), lr=1e-3)
    probe_crit  = nn.CrossEntropyLoss()

    es_s2 = EarlyStopping(patience=10, mode='max',
                          save_path=f'results/supcon_classifier{tag}.pth')

    probe_losses = []
    for epoch in range(1, PROBE_EPOCHS + 1):
        encoder.eval(); classifier.train()
        epoch_loss = 0.0
        for imgs, labels in ce_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = encoder(imgs)
            loss = probe_crit(classifier(feat), labels)
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
            epoch_loss += loss.item()
        probe_losses.append(epoch_loss / len(ce_loader))
        acc = evaluate_linear(encoder, classifier, test_loader, device)
        stop = es_s2.step(acc, classifier)

        print(f'[{label} S2] Epoch {epoch:2d}/{PROBE_EPOCHS} | Loss: {probe_losses[-1]:.4f} | '
              f'Test Acc: {acc:.2f}% | Best: {es_s2.best:.2f}% | Patience: {es_s2.counter}/{es_s2.patience}')

        if stop:
            print(f'[{label} S2] Early stopping at epoch {epoch}.')
            break

    es_s2.restore(classifier)

    # unfreeze (feature 추출용)
    for p in encoder.parameters():
        p.requires_grad = True

    print(f'\n[{label}] Best Test Accuracy: {es_s2.best:.2f}%')
    return encoder, classifier, es_s2.best, stage1_losses, probe_losses


def train_ablation(supcon_trainset, ce_loader, test_loader, device, config_key, config_vals):
    """Temperature 또는 Batch Size ablation 공통 함수"""
    results = {}
    for val in config_vals:
        if config_key == 'temperature':
            tau, bs = val, BATCH_SIZE
        else:
            tau, bs = TEMPERATURE, val

        tag = f'_{config_key}{val}'
        print(f'\n----- Ablation: {config_key}={val} -----')
        set_seed(42)

        abl_loader = DataLoader(supcon_trainset, batch_size=bs, shuffle=True,
                                num_workers=8, pin_memory=True, drop_last=True)

        enc  = ResNet18CIFAR().to(device)
        proj = ProjectionHead().to(device)
        crit = SupConLoss(temperature=tau)
        abl_lr = 0.05 * (bs / 256)
        opt  = optim.SGD(list(enc.parameters()) + list(proj.parameters()),
                         lr=abl_lr, momentum=0.9, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ABL_EPOCHS)

        es_abl = EarlyStopping(patience=50, mode='min', delta=1e-6)

        losses = []
        for epoch in range(1, ABL_EPOCHS + 1):
            enc.train(); proj.train()
            epoch_loss = 0.0
            for (img1, img2), labels in abl_loader:
                img1, img2 = img1.to(device), img2.to(device)
                labels = labels.to(device)
                opt.zero_grad()
                z1 = proj(enc(img1))
                z2 = proj(enc(img2))
                z = torch.stack([z1, z2], dim=1)  # (B, 2, 128)
                loss = crit(z, labels)
                loss.backward(); opt.step()
                epoch_loss += loss.item()
            sched.step()
            avg_loss = epoch_loss / len(abl_loader)
            losses.append(avg_loss)
            stop = es_abl.step(avg_loss, enc, proj)
            if epoch % 10 == 0:
                print(f'  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Patience: {es_abl.counter}/{es_abl.patience}')
            if stop:
                print(f'  Early stopping at epoch {epoch}.')
                break

        es_abl.restore(enc, proj)

        for p in enc.parameters():
            p.requires_grad = False
        clf  = LinearClassifier().to(device)
        p_opt = optim.Adam(clf.parameters(), lr=1e-3)
        p_crit = nn.CrossEntropyLoss()
        es_probe = EarlyStopping(patience=8, mode='max')
        for epoch in range(1, ABL_PROBE_EPOCHS + 1):
            enc.eval(); clf.train()
            for imgs, labels in ce_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    feat = enc(imgs)
                l = p_crit(clf(feat), labels)
                p_opt.zero_grad(); l.backward(); p_opt.step()
            acc = evaluate_linear(enc, clf, test_loader, device)
            if es_probe.step(acc, clf):
                break
        es_probe.restore(clf)
        best_acc = es_probe.best

        results[val] = {'acc': best_acc, 'losses': losses}
        print(f'  {config_key}={val} -> Best Acc: {best_acc:.2f}%')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. 시각화
# ══════════════════════════════════════════════════════════════════════════════
def plot_tsne_comparison(ce_feats, sc_feats, labels, n_samples=5000):
    idx = np.random.choice(len(ce_feats), min(n_samples, len(ce_feats)), replace=False)
    lbl = labels[idx]

    print('t-SNE 계산 중 (CE) ...')
    ce_emb = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42).fit_transform(ce_feats[idx])
    print('t-SNE 계산 중 (SupCon) ...')
    sc_emb = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42).fit_transform(sc_feats[idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = cm.get_cmap('tab10', 10)
    for ax, emb, title in zip(axes, [ce_emb, sc_emb], ['Cross-Entropy', 'SupCon']):
        for c in range(10):
            m = lbl == c
            ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(c)],
                       label=CIFAR10_CLASSES[c], s=6, alpha=0.7)
        ax.set_title(f't-SNE: {title}', fontsize=14)
        ax.legend(markerscale=3, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('results/tsne_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/tsne_comparison.png')


def plot_ablation(results, config_key, save_path):
    vals = list(results.keys())
    accs = [results[v]['acc'] for v in vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color = 'steelblue' if config_key == 'temperature' else 'tomato'

    axes[0].bar([str(v) for v in vals], accs, color=color, edgecolor='black')
    axes[0].set_xlabel(config_key.replace('_', ' ').title())
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title(f'Accuracy vs {config_key}')
    for i, a in enumerate(accs):
        axes[0].text(i, a + 0.1, f'{a:.2f}', ha='center', fontsize=10)

    for v in vals:
        axes[1].plot(results[v]['losses'], label=f'{config_key}={v}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SupCon Loss')
    axes[1].set_title(f'Loss Curves by {config_key}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_loss_curves(ce_losses, sc_losses, probe_losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ce_losses, color='steelblue', label='CE Loss')
    axes[0].set_title('Cross-Entropy Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(sc_losses, color='tomato', label='SupCon Loss (Stage1)')
    axes[1].plot(range(SUPCON_EPOCHS, SUPCON_EPOCHS + len(probe_losses)),
                 probe_losses, color='orange', linestyle='--', label='Probe CE Loss (Stage2)')
    axes[1].set_title('SupCon Training Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/loss_curves.png')


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    supcon_loader, ce_loader, test_loader, embed_loader = build_dataloaders(BATCH_SIZE)

    # ─── (1) CE Baseline ────────────────────────────────────────────────────
    ce_model, ce_acc, ce_losses = train_ce(ce_loader, test_loader, DEVICE)

    # ─── (2) SupCon ─────────────────────────────────────────────────────────
    sc_encoder, sc_classifier, sc_acc, sc_losses, probe_losses = train_supcon(
        supcon_loader, ce_loader, test_loader, DEVICE,
        temperature=TEMPERATURE, batch_size=BATCH_SIZE
    )

    # ─── (3) Loss 곡선 ─────────────────────────────────────────────────────
    plot_loss_curves(ce_losses, sc_losses, probe_losses)

    # ─── (4) t-SNE ──────────────────────────────────────────────────────────
    print('\n[t-SNE] Feature 추출 중 ...')
    ce_feats, ce_labels = extract_features(ce_model.encoder, embed_loader, DEVICE)
    sc_feats, sc_labels = extract_features(sc_encoder, embed_loader, DEVICE)

    plot_tsne(ce_feats, ce_labels,
              title='t-SNE: Cross-Entropy Representation',
              save_path='results/ce_tsne.png')
    plot_tsne(sc_feats, sc_labels,
              title='t-SNE: SupCon Representation',
              save_path='results/supcon_tsne.png')
    plot_tsne_comparison(ce_feats, sc_feats, ce_labels)

    # ─── (5) Temperature Ablation ───────────────────────────────────────────
    print('\n' + '='*55)
    print('  Ablation: Temperature')
    print('='*55)
    # ablation용 trainset 직접 접근
    supcon_trainset = supcon_loader.dataset
    temp_results = train_ablation(supcon_trainset, ce_loader, test_loader,
                                  DEVICE, 'temperature', TEMP_LIST)
    plot_ablation(temp_results, 'temperature', 'results/temperature_ablation.png')

    # ─── (6) Batch Size Ablation ────────────────────────────────────────────
    print('\n' + '='*55)
    print('  Ablation: Batch Size')
    print('='*55)
    bs_results = train_ablation(supcon_trainset, ce_loader, test_loader,
                                DEVICE, 'batch_size', BATCH_LIST)
    plot_ablation(bs_results, 'batch_size', 'results/batchsize_ablation.png')

    # ─── (7) 결과 요약 ──────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('            전체 실험 결과 요약')
    print('='*55)
    print(f'Cross-Entropy (100 epoch):          {ce_acc:.2f}%')
    print(f'SupCon + Linear Probe (100+20):     {sc_acc:.2f}%')
    print()
    print('--- Temperature Ablation ---')
    for tau, v in temp_results.items():
        print(f'  τ={tau:<5}: {v["acc"]:.2f}%')
    print()
    print('--- Batch Size Ablation ---')
    for bs, v in bs_results.items():
        print(f'  bs={bs:<4}: {v["acc"]:.2f}%')
    print('='*55)

    rows = [
        {'Experiment': 'CE Baseline',  'Config': 'default',          'Accuracy': ce_acc},
        {'Experiment': 'SupCon',       'Config': f'tau={TEMPERATURE}, bs={BATCH_SIZE}', 'Accuracy': sc_acc},
    ]
    for tau, v in temp_results.items():
        rows.append({'Experiment': 'SupCon (temp ablation)',  'Config': f'tau={tau}', 'Accuracy': v['acc']})
    for bs, v in bs_results.items():
        rows.append({'Experiment': 'SupCon (bs ablation)', 'Config': f'bs={bs}',  'Accuracy': v['acc']})

    pd.DataFrame(rows).to_csv('results/full_summary.csv', index=False)
    print('\n결과 저장 완료: results/full_summary.csv')
    print('생성된 이미지:')
    for f in sorted(os.listdir('results')):
        if f.endswith('.png'):
            print(f'  results/{f}')


if __name__ == '__main__':
    main()
