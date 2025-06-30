import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, average_precision_score
from model.model1 import MoRFPredictionBranch1
from model.model2 import MoRFPredictionBranch2
from model.FusionModel import FusionModel


class FocalLoss(nn.Module):
    def __init__(self, initial_alpha=0.8, gamma=2.0, label_smoothing=0.1, weight=None, temperature=1.0, grad_clip=None,
                 alpha_step=0.01, patience=5):
        super().__init__()
        self.alpha = initial_alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.alpha_step = alpha_step
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def forward(self, inputs, targets, mask_valid):
        # Temperature scaling
        inputs = inputs / self.temperature

        # Label smoothing
        num_classes = inputs.size(1)
        smooth_targets = (1 - self.label_smoothing) * targets + self.label_smoothing / num_classes

        # Binary Cross Entropy Loss with logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, smooth_targets, reduction='none', weight=self.weight)

        # Focal Loss calculation
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Mask valid entries and compute the mean loss
        masked_F_loss = F_loss * mask_valid
        loss = masked_F_loss.sum() / mask_valid.sum()

        # Gradient clipping if enabled
        if self.grad_clip is not None:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(inputs, self.grad_clip)

        # Update best loss and counter for dynamic alpha adjustment
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.alpha = min(self.alpha + self.alpha_step, 1.0)
                self.counter = 0

        return loss

# ----------- 数据加载函数 -----------
def read_labels_from_fasta(fasta_file, max_len=300):
    labels = []
    lengths = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                label_match = re.search(r'>([01]+)', line)
                if not label_match: continue
                label_str = label_match.group(1)
                original_length = len(label_str)
                current_label = label_str[:max_len] if original_length > max_len else label_str.ljust(max_len, '0')
                labels.append([int(c) for c in current_label])
                lengths.append(min(original_length, max_len))
    return torch.tensor(labels, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int32)


# ----------- 数据增强 -----------
def mask_features(x, mask_prob=0.1):
    mask = torch.rand_like(x) < mask_prob
    return x.masked_fill(mask, 0)


# ----------- 早停类 -----------
class EarlyStopping:
    def __init__(self, patience, verbose):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, models, optimizer):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                'model1': models[0].state_dict(),
                'model2': models[1].state_dict(),
                'fusion_model': models[2].state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if self.verbose: print(f"Best model updated with val_loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose: print(f"Early stopping triggered at validation loss: {val_loss:.4f}")


# ----------- 主训练流程 -----------
if __name__ == "__main__":
    # 加载数据
    esm2_features = np.load('/home/ys/sunhuaiyang/predict/feature/train/esm2_features.npy')
    t5_features = np.load('/home/ys/sunhuaiyang/predict/feature/train/t5_features.npy')
    #pssm_features = np.load('/home/ys/sunhuaiyang/predict_morf/Features/train/pssm_features.npy')
    #merged_features_model1 = np.concatenate([t5_features, pssm_features], axis=2)  # 结果形状 (n, 300, 1044)

    # 转换为PyTorch张量
    #merged_features_model1 = torch.tensor(merged_features_model1, dtype=torch.float32)  # 形状 (n, 300, 1044)
    merged_features_model1 = torch.tensor(t5_features, dtype=torch.float32)  # 形状 (n, 300, 1024)
    merged_features_model2 = torch.tensor(esm2_features, dtype=torch.float32)  # 形状 (n, 300, 1280)
    labels, lengths = read_labels_from_fasta('/home/ys/sunhuaiyang/predict/data/train.fasta')

    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_save_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(merged_features_model1)):
        print(f"\nTraining fold {fold + 1}/5")

        # 数据分割
        X_train_m1, X_val_m1 = merged_features_model1[train_idx], merged_features_model1[val_idx]
        X_train_m2, X_val_m2 = merged_features_model2[train_idx], merged_features_model2[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_lengths, val_lengths = lengths[train_idx], lengths[val_idx]

        # 创建DataLoader
        train_dataset = TensorDataset(X_train_m1, X_train_m2, y_train, train_lengths)
        val_dataset = TensorDataset(X_val_m1, X_val_m2, y_val, val_lengths)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #model1 = MoRFPredictionBranch1(feat_dim=1044).to(device)
        model1 = MoRFPredictionBranch1(feat_dim=1024).to(device)
        model2 = MoRFPredictionBranch2(input_dim=1280).to(device)
        fusion = FusionModel().to(device)

        # 定义优化器
        optimizer = optim.AdamW(
            list(model1.parameters()) + list(model2.parameters()) + list(fusion.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        # 训练循环
        for epoch in range(100):
            model1.train()
            model2.train()
            fusion.train()
            running_loss = 0.0
            total_samples = 0

            # 训练批次
            for batch in train_loader:
                features1, features2, targets, lens = batch
                features1 = mask_features(features1.to(device))  # 数据增强
                features2 = mask_features(features2.to(device))
                targets = targets.to(device)
                lens = lens.to(device)
                B, T = features1.size(0), features1.size(1)

                # 生成注意力掩码
                mask_valid = (torch.arange(T, device=device).unsqueeze(0) < lens.unsqueeze(1)).float()

                optimizer.zero_grad()

                # 前向传播
                output1 = model1(features1, lengths=lens)
                output2 = model2(features2)
                final_output = fusion(output1, output2)

                # 计算损失
                loss = FocalLoss()(final_output, targets, mask_valid)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), 0.5)
                optimizer.step()

                running_loss += loss.item() * mask_valid.sum().item()
                total_samples += mask_valid.sum().item()

            # 学习率调度
            scheduler.step()

            # 计算平均训练损失
            avg_train_loss = running_loss / total_samples if total_samples > 0 else 0
            print(f'Epoch {epoch + 1}/100 Train Loss: {avg_train_loss:.4f}')

            # 验证阶段
            model1.eval()
            model2.eval()
            fusion.eval()
            val_loss = 0.0
            all_labels = []
            all_probs = []
            total_val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    features1, features2, targets, lens = batch
                    features1 = features1.to(device)
                    features2 = features2.to(device)
                    targets = targets.to(device)
                    lens = lens.to(device)
                    B, T = features1.size(0), features1.size(1)

                    # 生成掩码
                    mask_valid = (torch.arange(T, device=device).unsqueeze(0) < lens.unsqueeze(1)).float()

                    # 前向传播
                    output1 = model1(features1, lengths=lens)
                    output2 = model2(features2)
                    final_output = fusion(output1, output2)

                    # 计算损失
                    loss = FocalLoss()(final_output, targets, mask_valid)
                    val_loss += loss.item() * mask_valid.sum().item()

                    # 收集预测结果
                    probs = torch.sigmoid(final_output).cpu().numpy()
                    mask_valid_np = mask_valid.cpu().numpy().astype(bool)
                    for b in range(probs.shape[0]):
                        valid_indices = mask_valid_np[b]
                        all_labels.extend(targets[b][valid_indices].cpu().numpy().flatten())
                        all_probs.extend(probs[b][valid_indices].flatten())
                    total_val_samples += mask_valid.sum().item()

            # 计算验证指标
            avg_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            all_preds = np.round(all_probs)

            # 指标计算（处理全正/负样本情况）
            if len(np.unique(all_labels)) > 1:
                val_auc = roc_auc_score(all_labels, all_probs)
                val_ap = average_precision_score(all_labels, all_probs)
                val_mcc = matthews_corrcoef(all_labels, all_preds)
                cm = confusion_matrix(all_labels, all_preds)
                if cm.size == 1:
                    tn = fp = fn = tp = 0
                    if all_preds.all() == 1:
                        tp = cm[0, 0]
                    else:
                        tn = cm[0, 0]
                else:
                    tn, fp, fn, tp = cm.ravel()
            else:
                val_auc = val_ap = val_mcc = 0.5
                tn = fp = fn = tp = 0

            val_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            val_bacc = (val_tpr + (1 - val_fpr)) / 2

            print(f'Val Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | AP: {val_ap:.4f}')
            print(f'TPR: {val_tpr:.4f} | FPR: {val_fpr:.4f} | BACC: {val_bacc:.4f} | MCC: {val_mcc:.4f}')

            # 早停检查
            early_stopping(avg_val_loss, [model1, model2, fusion], optimizer)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # 保存最佳模型
        if early_stopping.best_state is not None:
            save_path = f'/home/ys/sunhuaiyang/predict/saved_models/fold{fold + 1}_best.pth'
            torch.save(early_stopping.best_state, save_path)
            model_save_paths.append(save_path)
        else:
            save_path = f'/home/ys/sunhuaiyang/predict/saved_models/fold{fold + 1}_final.pth'
            torch.save({
                'model1': model1.state_dict(),
                'model2': model2.state_dict(),
                'fusion': fusion.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            model_save_paths.append(save_path)
        print(f"Fold {fold + 1} model saved to {save_path}")

    print("Training completed. Models saved at:")
    for path in model_save_paths:
        print(path)
