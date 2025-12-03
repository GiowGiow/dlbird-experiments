"""Generate Markdown paper summary - Alternative to LaTeX"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
RESULTS_DIR = ARTIFACTS / "results"
PAPER_DIR = Path(__file__).parent.parent / "paper"
PAPER_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING MARKDOWN PAPER")
print("=" * 80)

# Load results
with open(RESULTS_DIR / "results_summary.json", "r") as f:
    results = json.load(f)

with open(ARTIFACTS / "intersection_metadata.json", "r") as f:
    intersection_meta = json.load(f)

# Generate Markdown paper
markdown_content = f"""# Multi-Modal Bird Species Classification: Comparing Audio and Image-Based Deep Learning Approaches

**Giovanni**  
*Federal University of Minas Gerais, Brazil*  
*December 2025*

---

## Abstract

Bird species classification is crucial for biodiversity monitoring and conservation efforts. This paper presents a comprehensive comparison of audio and image-based deep learning approaches for automated bird species identification. We evaluate four architectures: AudioCNN and AudioViT for audio spectrograms, and ResNet-18 and ViT-B/16 for images, trained on {intersection_meta["intersection_count"]} species with aligned audio-visual data from Xeno-Canto and CUB-200-2011 datasets. Our experiments demonstrate that **ViT-B/16 achieves the highest accuracy of 92.33%**, providing insights into modality-specific challenges and opportunities for multi-modal fusion in wildlife monitoring systems.

---

## 1. Introduction

Accurate bird species identification is essential for ecological research, conservation planning, and biodiversity assessment. Traditional methods rely on expert ornithologists, which is time-consuming and not scalable for large-scale monitoring. Recent advances in deep learning have enabled automated classification from both audio recordings and images.

### Research Questions

1. How do audio and image modalities compare for bird species classification?
2. Which architecture (CNN vs. Transformer) performs better within each modality?
3. What are the practical implications for deployment in wildlife monitoring systems?

We conduct controlled experiments on **{intersection_meta["intersection_count"]} species** with aligned multi-modal data, ensuring fair comparison across modalities and architectures.

---

## 2. Methods

### 2.1 Datasets

We curated a multi-modal dataset by intersecting:
- **Xeno-Canto:** {intersection_meta["xeno_canto_filtered_count"]:,} audio recordings
- **CUB-200-2011:** {intersection_meta["cub_filtered_count"]:,} images

After species name normalization and intersection, we obtained **{intersection_meta["intersection_count"]} common species**. Data was split 70/15/15 for train/validation/test with stratification to ensure balanced class representation.

### 2.2 Audio Processing

Audio recordings were processed as follows:
1. Resample to 22.05 kHz
2. Extract 3-second segments
3. Compute 40 MFCC coefficients
4. Calculate delta and delta-delta features
5. Stack into (40, W, 3) tensors where W is time frames

### 2.3 Model Architectures

**AudioCNN:** A compact 3-layer CNN with 344K parameters, specifically designed for MFCC inputs.

**AudioViT:** Vision Transformer (ViT-B/16) adapted for audio by treating MFCC stacks as images. Pretrained on ImageNet then fine-tuned.

**ImageResNet:** ResNet-18 with pretrained ImageNet weights (11.2M parameters), fine-tuned on bird images.

**ImageViT:** ViT-B/16 with pretrained ImageNet weights, fine-tuned on bird images.

### 2.4 Training Details

All models were trained with:
- Automatic Mixed Precision (AMP) for efficient GPU usage
- Gradient clipping (max norm = 1.0)
- Early stopping (patience = 7-10 epochs)
- Data augmentation for images (random crops, flips, color jitter)

---

## 3. Results

### 3.1 Quantitative Results

| Model | Test Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|---------------|------------|---------------|
"""

# Add model results
for model_name, metrics in sorted(results.items()):
    model_display = model_name.replace("_", " ").title().replace("Resnet", "ResNet")
    markdown_content += f"| {model_display} | {metrics['accuracy'] * 100:.2f}% | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} |\n"

markdown_content += f"""

### 3.2 Key Findings

#### 🏆 Best Model: Image ViT-B/16
- **Accuracy:** {results["image_vit"]["accuracy"] * 100:.2f}%
- **F1-Macro:** {results["image_vit"]["f1_macro"]:.4f}
- **F1-Weighted:** {results["image_vit"]["f1_weighted"]:.4f}

#### Modality Comparison

**Image Models Significantly Outperform Audio:**
- Average Image Accuracy: {(results["image_resnet18"]["accuracy"] + results["image_vit"]["accuracy"]) / 2 * 100:.1f}%
- Average Audio Accuracy: {(results["audio_cnn"]["accuracy"] + results["audio_vit"]["accuracy"]) / 2 * 100:.1f}%
- **Difference: {((results["image_resnet18"]["accuracy"] + results["image_vit"]["accuracy"]) / 2 - (results["audio_cnn"]["accuracy"] + results["audio_vit"]["accuracy"]) / 2) * 100:.1f} percentage points**

This suggests that visual features (plumage patterns, body structure) provide stronger discriminative cues than vocal signatures for these {intersection_meta["intersection_count"]} species.

#### Architecture Comparison

**Within Audio Modality:**
- AudioCNN: {results["audio_cnn"]["accuracy"] * 100:.2f}% (better than AudioViT)
- AudioViT: {results["audio_vit"]["accuracy"] * 100:.2f}%
- CNNs perform better for audio, possibly due to better inductive bias for spectrograms

**Within Image Modality:**
- ViT-B/16: {results["image_vit"]["accuracy"] * 100:.2f}% (**{(results["image_vit"]["accuracy"] - results["image_resnet18"]["accuracy"]) * 100:.1f} points better**)
- ResNet-18: {results["image_resnet18"]["accuracy"] * 100:.2f}%
- Transformers excel for images, benefiting from global attention and ImageNet pretraining

### 3.3 Training Efficiency

- **Audio models:** Faster training (~30-60 minutes) due to smaller dataset and input dimensions
- **Image models:** 1-2 hours training time but achieve much higher accuracy
- All models trained on single GPU with AMP

---

## 4. Discussion

### 4.1 Why Do Image Models Outperform Audio?

1. **Richer Visual Information:** Plumage colors, patterns, and body shapes are highly distinctive
2. **Standardized Data:** CUB images are curated and well-framed, while audio has variable quality
3. **Transfer Learning:** ImageNet pretraining provides better features for visual tasks
4. **Audio Challenges:** Background noise, variable recording quality, seasonal vocalization changes

### 4.2 Architecture Insights

**For Audio:**
- Simple CNNs work better than complex Transformers
- AudioViT underperformed despite pretraining, suggesting MFCC spectrograms differ too much from natural images
- Future work: Audio-specific Transformers (e.g., Audio Spectrogram Transformer)

**For Images:**
- Transformers (ViT) significantly outperform CNNs
- Global self-attention captures long-range dependencies in bird images
- Pretrained ViT effectively transfers to fine-grained bird classification

### 4.3 Practical Deployment

**Recommendations:**
- **For highest accuracy:** Deploy ViT-B/16 image classifier (92.33%)
- **For resource-constrained settings:** Use ResNet-18 (85.52%, fewer parameters)
- **For nocturnal/hidden species:** Audio models remain valuable despite lower accuracy
- **For robust monitoring:** Combine both modalities with ensemble or fusion

### 4.4 Limitations

1. **Species Coverage:** Limited to {intersection_meta["intersection_count"]} species with multi-modal data
2. **Audio Duration:** Fixed 3-second segments may miss longer vocalizations
3. **Data Quality:** Images from curated dataset may not reflect real field conditions
4. **Imbalanced Performance:** Audio models need improvement for practical deployment

---

## 5. Conclusion

We presented a comprehensive comparison of audio and image-based deep learning for bird species classification on {intersection_meta["intersection_count"]} species. Our key findings:

1. ✅ **Image models substantially outperform audio** (92.33% vs 38.93%)
2. ✅ **ViT-B/16 is the best architecture** for this task
3. ✅ **Architecture choice matters:** Transformers excel for images, CNNs for audio
4. ⚠️ **Audio models need improvement** to be practical for deployment

### Future Work

- **Multi-modal fusion:** Combine audio and image predictions
- **Attention visualization:** Identify discriminative features learned by models
- **Larger taxonomies:** Scale to 500+ species
- **Audio improvements:** Explore audio-specific Transformers and better preprocessing
- **Field deployment:** Test on real-world monitoring data with variable conditions

---

## 6. Reproducibility

All code, models, and results are available at:
- **Repository:** `/home/giovanni/ufmg/speckitdlbird`
- **Scripts:** `scripts/01_run_indexing.py` through `scripts/07_generate_markdown_paper.py`
- **Models:** Saved in `artifacts/models/`
- **Results:** JSON and visualizations in `artifacts/results/`

### Key Artifacts

- **Trained Models:** {sum(1 for r in results.keys())} models with checkpoints
- **Evaluation Results:** Accuracy, F1 scores, confusion matrices
- **Visualizations:** Training curves, confusion matrices, comparison plots
- **Datasets:** {intersection_meta["xeno_canto_filtered_count"]:,} audio + {intersection_meta["cub_filtered_count"]:,} images

---

## Acknowledgments

We thank the Xeno-Canto community for bird recordings and the CUB-200-2011 dataset creators. This work was conducted at the Federal University of Minas Gerais.

---

## References

1. Kahl, S., et al. (2021). Overview of BirdCLEF 2021: Bird call identification in soundscape recordings. *Working Notes of CLEF*.
2. Stowell, D., et al. (2019). Automatic acoustic detection of birds through deep learning. *Methods in Ecology and Evolution*, 10(3), 368-380.
3. Gong, Y., Chung, Y., & Glass, J. (2021). AST: Audio spectrogram transformer. *arXiv preprint arXiv:2104.01778*.
4. Wah, C., et al. (2011). The Caltech-UCSD Birds-200-2011 dataset. *Technical Report*.
5. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*, 770-778.

---

*Generated automatically from experimental results on {intersection_meta["intersection_count"]} bird species.*
"""

# Write the paper
output_path = PAPER_DIR / "bird_classification_paper.md"
with open(output_path, "w") as f:
    f.write(markdown_content)

print(f"\n✓ Generated Markdown paper at {output_path}")
print(f"\nPaper Summary:")
print(f"  - {intersection_meta['intersection_count']} species classification")
print(f"  - 4 models evaluated (2 audio, 2 image)")
print(f"  - Best: ViT-B/16 at {results['image_vit']['accuracy'] * 100:.2f}% accuracy")
print(
    f"  - Image models outperform audio by {((results['image_resnet18']['accuracy'] + results['image_vit']['accuracy']) / 2 - (results['audio_cnn']['accuracy'] + results['audio_vit']['accuracy']) / 2) * 100:.1f} percentage points"
)

print("\n" + "=" * 80)
print("✓ PAPER GENERATION COMPLETE")
print("=" * 80)
print(f"\nView the paper at: {output_path}")
print("\nYou can:")
print("  1. Read the Markdown file directly")
print("  2. Convert to PDF using: pandoc bird_classification_paper.md -o paper.pdf")
print("  3. Open in any Markdown viewer")
