import re
import numpy as np
import torch
import warnings
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, average_precision_score
from model.model1 import MoRFPredictionBranch1
from model.model2 import MoRFPredictionBranch2
from model.FusionModel import FusionModel


def read_labels_from_fasta(fasta_file, max_len=300):
    labels = []
    lengths = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                label_match = re.search(r'>([01]+)', line)
                if not label_match:
                    continue
                label_str = label_match.group(1)
                original_length = len(label_str)
                if original_length > max_len:
                    current_label = label_str[:max_len]
                    current_length = max_len
                else:
                    current_label = label_str.ljust(max_len, '0')
                    current_length = original_length
                labels.append([int(c) for c in current_label])
                lengths.append(current_length)
    return torch.tensor(labels, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

esm2_test = np.load('/home/ys/sunhuaiyang/predict/feature/test/test1_esm2_features.npy')
t5_test = np.load('/home/ys/sunhuaiyang/predict/feature/test/test1_T5_features.npy')

labels, lengths = read_labels_from_fasta('/home/ys/sunhuaiyang/predict/data/test1.fasta')

esm2_tensor = torch.tensor(esm2_test, dtype=torch.float32)
merged_tensor = torch.tensor(t5_test, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)
lengths_tensor = torch.tensor(lengths, dtype=torch.int32)

test_dataset = TensorDataset(merged_tensor, esm2_tensor, labels_tensor, lengths_tensor)

fold_metrics = {
    'fold': [], 'accuracy': [], 'auc': [], 'mcc': [],
    'tpr': [], 'fpr': [], 'bacc': [], 'f1': [], 'ap': []
}

for fold_idx in range(5):
    print(f"\nEvaluating Fold {fold_idx + 1}/5")

    model_path = f'/home/ys/sunhuaiyang/predict/saved_models/fold{fold_idx + 1}_best.pth'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        checkpoint = torch.load(model_path, map_location=device)

    model1 = MoRFPredictionBranch1(feat_dim=1024).to(device)
    model2 = MoRFPredictionBranch2(input_dim=1280).to(device)
    fusion = FusionModel().to(device)

    model1.load_state_dict(checkpoint['model1'])
    model2.load_state_dict({k: v for k, v in checkpoint['model2'].items()
                            if not k.startswith('masking.')}, strict=False)
    fusion.load_state_dict(checkpoint['fusion_model'])

    model1.eval()
    model2.eval()
    fusion.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_probs, all_labels, all_lengths = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            features_m1, features_m2, labels, lengths = batch
            features_m1 = features_m1.to(device)
            features_m2 = features_m2.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output1 = model1(features_m1, lengths=lengths)
            output2 = model2(features_m2)
            logits = fusion(output1, output2)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_lengths.append(lengths.cpu().numpy())

    final_probs, final_labels = [], []
    for probs, labels, lens in zip(all_probs, all_labels, all_lengths):
        for i in range(len(lens)):
            seq_len = int(lens[i])
            final_probs.extend(probs[i][:seq_len])
            final_labels.extend(labels[i][:seq_len])

    final_probs = np.array(final_probs)
    final_labels = np.array(final_labels)
    final_preds = (final_probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(final_labels, final_probs)
    except ValueError:
        auc = 0.5

    try:
        mcc = matthews_corrcoef(final_labels, final_preds)
    except:
        mcc = 0.0

    tn, fp, fn, tp = confusion_matrix(final_labels, final_preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    bacc = (tpr + (1 - fpr)) / 2
    accuracy = (final_preds == final_labels).mean()

    try:
        ap = average_precision_score(final_labels, final_probs)
    except:
        ap = 0.0

    fold_metrics['fold'].append(fold_idx + 1)
    fold_metrics['accuracy'].append(accuracy)
    fold_metrics['auc'].append(auc)
    fold_metrics['mcc'].append(mcc)
    fold_metrics['tpr'].append(tpr)
    fold_metrics['fpr'].append(fpr)
    fold_metrics['bacc'].append(bacc)
    fold_metrics['f1'].append(f1)
    fold_metrics['ap'].append(ap)

    saved_results = []
    for i in range(len(final_labels)):
        saved_results.append({
            'true_label': final_labels[i],
            'pred_prob': final_probs[i]
        })

    save_path = f'/home/ys/sunhuaiyang/predict/save_model/saved_model.csv' 
    results_df = pd.DataFrame(saved_results)
    results_df.to_csv(save_path, index=False)
    print(f"Predictions for Fold {fold_idx + 1} saved to '{save_path}'.")

    print(f"Fold {fold_idx + 1} Metrics:")
    print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")
    print(f"TPR: {tpr:.4f} | FPR: {fpr:.4f} | BACC: {bacc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

print("\nSummary of All Folds:")
print("Fold | Accuracy | AUC    | MCC    | TPR    | FPR    | BACC   | F1    | AP   ")
for i in range(5):
    print(f"{fold_metrics['fold'][i]:4} | "
          f"{fold_metrics['accuracy'][i]:.4f} | "
          f"{fold_metrics['auc'][i]:.4f} | "
          f"{fold_metrics['mcc'][i]:.4f} | "
          f"{fold_metrics['tpr'][i]:.4f} | "
          f"{fold_metrics['fpr'][i]:.4f} | "
          f"{fold_metrics['bacc'][i]:.4f} | "
          f"{fold_metrics['f1'][i]:.4f} | "
          f"{fold_metrics['ap'][i]:.4f}")  

print("\nAverage Metrics:")
print(f"Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"AUC:      {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
print(f"MCC:      {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
print(f"TPR:      {np.mean(fold_metrics['tpr']):.4f} ± {np.std(fold_metrics['tpr']):.4f}")
print(f"FPR:      {np.mean(fold_metrics['fpr']):.4f} ± {np.std(fold_metrics['fpr']):.4f}")
print(f"BACC:     {np.mean(fold_metrics['bacc']):.4f} ± {np.std(fold_metrics['bacc']):.4f}")
print(f"F1:       {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"AP:       {np.mean(fold_metrics['ap']):.4f} ± {np.std(fold_metrics['ap']):.4f}") 

