"""Evaluate AST model on test set."""

import sys
from pathlib import Path
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_ast import AudioAST
from src.datasets.audio_spectrogram import AudioSpectrogramDataset, collate_spectrograms

def calculate_map(y_true, y_probs, num_classes):
    """Calculate mean Average Precision."""
    aps = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_score_c = y_probs[:, c]
        
        # Sort by predicted probability
        sorted_indices = np.argsort(-y_score_c)
        y_true_sorted = y_true_c[sorted_indices]
        
        # Calculate precision at each position
        cum_tp = np.cumsum(y_true_sorted)
        cum_fp = np.cumsum(1 - y_true_sorted)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
        
        # Average precision: mean of precisions at relevant positions
        ap = np.sum(precisions * y_true_sorted) / (np.sum(y_true_sorted) + 1e-10)
        aps.append(ap)
    
    return np.mean(aps)

def evaluate_model(args):
    """Evaluate AST model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load splits
    with open(args.splits_path, 'r') as f:
        splits = json.load(f)
    
    # Load dataframe for species mapping
    artifacts_dir = Path(args.splits_path).parent.parent
    xc_df = pd.read_parquet(artifacts_dir / "xeno_canto_filtered.parquet")
    xc_counts = xc_df["species_normalized"].value_counts()
    species_to_keep = xc_counts[xc_counts >= 2].index
    xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()
    species_list = sorted(xc_df["species_normalized"].unique())
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    num_classes = len(species_list)
    
    # Create dataset
    dataset = AudioSpectrogramDataset(
        df=xc_df,
        cache_dir=Path(args.cache_dir),
        split=splits['test'],
        species_to_idx=species_to_idx,
        augment=None
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_spectrograms
    )
    
    print(f"\nTest set: {len(dataset)} samples, {num_classes} species")
    
    # Load model
    model = AudioAST(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size}/{len(dataset)} samples...")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    mAP = calculate_map(all_labels, all_probs, num_classes)
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"mAP:           {mAP:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'checkpoint': str(args.checkpoint),
        'epoch': checkpoint['epoch'],
        'val_acc': float(checkpoint.get('val_acc', 0)),
        'test_acc': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'mAP': float(mAP)
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{args.output_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir / f'{args.output_name}_results.json'}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=species_list,
                yticklabels=species_list)
    plt.title(f'Confusion Matrix - {args.output_name}\nTest Accuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    cm_path = output_dir / f"{args.output_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    
    # Per-class F1 scores
    report = classification_report(all_labels, all_preds, 
                                   target_names=species_list,
                                   output_dict=True, zero_division=0)
    
    # Save per-class results
    per_class_results = []
    for species in splits['species_list']:
        if species in report:
            per_class_results.append({
                'species': species,
                'precision': report[species]['precision'],
                'recall': report[species]['recall'],
                'f1-score': report[species]['f1-score'],
                'support': report[species]['support']
            })
    
    with open(output_dir / f"{args.output_name}_per_class.json", 'w') as f:
        json.dump(per_class_results, f, indent=2)
    
    print(f"Per-class results saved to {output_dir / f'{args.output_name}_per_class.json'}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AST model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--splits-path', type=str, 
                       default='artifacts/splits/xeno_canto_audio_splits.json',
                       help='Path to splits JSON')
    parser.add_argument('--audio-dir', type=str,
                       default='data/xeno_canto/audio',
                       help='Path to audio directory')
    parser.add_argument('--cache-dir', type=str,
                       default='artifacts/audio_lms_cache/xeno_canto',
                       help='Path to LMS cache directory')
    parser.add_argument('--output-dir', type=str,
                       default='artifacts/results',
                       help='Directory to save results')
    parser.add_argument('--output-name', type=str, required=True,
                       help='Name for output files')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    evaluate_model(args)
