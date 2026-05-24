# **EnScaT-MIL: an artificial intelligence model for predicting** ***TP53*** **mutation from p53 immunohistochemistry in endometrial cancer**

This repository implements a **EnScaT-MIL** model for p53 classification on whole-slide images (WSIs). The model integrates text prompt embeddings with the TransMIL architecture for improved classification performance.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- h5py
- transformers
- nystrom-attention

Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib tqdm h5py transformers nystrom-attention
```

## Data Preprocess

we follow the CLAM's WSI processing solution (<https://github.com/mahmoodlab/CLAM>)

```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch

# Feature Extraction
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

### Training Script

The training script `train.py` implements 5-fold cross-validation training:

```bash
python train.py
```

### Training Configuration

Key parameters in the training script:

### Model Architecture

The PFMIL model consists of:

1. **Feature Encoder**: TransMIL with Nystrom attention
2. **Text Encoder**: BERT-based prompt encoder
3. **Classifier**: Two-layer MLP with dropout

```python
class SingleTaskSlideEncoder(nn.Module):
    def __init__(self, input_feature_dim=512):
        super().__init__()
        self.feature_encoder = TransMIL(n_classes=2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_feature_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
```

### Training Output

Models are saved to:

```
/result/
├── best_p53_model_fold1.pth
├── best_p53_model_fold2.pth
├── best_p53_model_fold3.pth
├── best_p53_model_fold4.pth
├── best_p53_model_fold5.pth
```

## Testing

### Testing Script

The testing script `test.py` evaluates the ensemble model:

```bash
python test.py
```

<br />

### Evaluation Metrics

The following metrics are computed with 95% confidence intervals:

| Metric      | Description                           |
| ----------- | ------------------------------------- |
| Accuracy    | Overall classification accuracy       |
| AUC         | Area under ROC curve                  |
| F1-Score    | Harmonic mean of precision and recall |
| Sensitivity | True positive rate (TPR)              |
| Specificity | True negative rate (TNR)              |
| PPV         | Positive predictive value             |
| NPV         | Negative predictive value             |

#
