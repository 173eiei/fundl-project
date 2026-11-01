# Car Parking Vision

A reproducible deep-learning capstone that classifies an individual parking-space image as **Empty** or **Occupied** using a CNN trained from scratch with PyTorch.

## Project status

**Implementation completed**
- [x] Repository structure, README, and Google Colab entry point
- [x] PKLot preprocessing pipeline with a maximum of 10,000 parking-space crops
- [x] Reproducible date-grouped split: approximately 70% train / 15% validation / 15% test
- [x] Majority-class baseline script
- [x] CNN training script with validation monitoring, early stopping, and learning-curve output
- [x] Held-out evaluation script with classification metrics, confusion matrix, and saved misclassified examples

**Experiment and submission still pending**
- [ ] Obtain a Colab runtime and download/inspect the PKLot archive
- [ ] Run preprocessing and verify the actual crop count, class balance, and split counts
- [ ] Run the baseline and record its actual validation metrics
- [ ] Train the CNN and save the real checkpoint and learning curves
- [ ] Run the held-out test evaluation and record actual test metrics
- [ ] Inspect real successful and failed predictions
- [ ] Complete the 4–8 page report and 3–6 minute demo

The code and Colab workflow are prepared, but the dataset pipeline, training, and evaluation have **not yet been executed successfully in Colab**. No model scores or experimental results are claimed until they are produced by a real run. The current split is grouped by capture date; the test set uses unseen dates across the available PKLot sites.

## Dataset

We use the public PKLot parking-lot dataset by Almeida et al. (2015). It contains 12,417 full parking-lot images and approximately 695,900 labelled parking-space instances, captured from three fixed camera views: UFPR04, UFPR05, and PUCPR, under sunny, cloudy/overcast, and rainy conditions.

- Original dataset page: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
- Archive: https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
- Archive mirror description: https://huggingface.co/datasets/teenygrad/pklot
- The dataset mirror reports CC BY 4.0. Check the license file included with the downloaded archive before redistribution.

Citation:

Almeida, P. R. L., Oliveira, L. S., Britto Jr., A. S., Silva Jr., E. J., & Koerich, A. L. (2015). PKLot—A robust dataset for parking lot classification. Expert Systems with Applications, 42(11), 4937–4949. https://doi.org/10.1016/j.eswa.2015.02.009

## Evaluation protocol

The official archive organizes scenes as `parking1a`, `parking1b`, and `parking2`; the preparation script maps these to UFPR04, UFPR05, and PUCPR respectively. It also accepts archives already organized under the canonical UFPR04/UFPR05/PUCPR names. The task uses individual parking-space crops and binary labels: 0 = Empty, 1 = Occupied.

The experiment uses no more than **10,000 parking-space crops in total** (or fewer if fewer valid crops are available). The selected images are split approximately **70% train / 15% validation / 15% test**. Splitting is grouped by capture date across all sites, so a date cannot appear in more than one split. Actual counts can differ slightly from 7,000 / 1,500 / 1,500 because each date must remain intact. The exact counts, class balance, weather distribution, site distribution, and number of unique dates are saved in `data/processed/preparation_summary.json`.

The test set contains unseen capture dates drawn from the available camera sites. This measures performance on new dates, not generalisation to a completely unseen parking lot. Do not move test records into train/validation or use test metrics to select hyperparameters.

## Repository layout

    fundl-project/
    ├── notebooks/
    │   └── parking_vision.ipynb
    ├── src/
    │   ├── data.py
    │   ├── prepare_data.py
    │   ├── model.py
    │   ├── baseline.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── data/        # local only; ignored by Git
    ├── outputs/     # local only; ignored by Git
    ├── reports/
    ├── README.md
    ├── requirements.txt
    └── .gitignore

## Run on Google Colab

1. Open the Colab notebook: https://colab.research.google.com/github/173eiei/fundl-project/blob/main/notebooks/parking_vision.ipynb
2. Choose Runtime → Change runtime type → T4 GPU when available.
3. Run the notebook from top to bottom. It clones this repository into the temporary Colab runtime, downloads PKLot, preprocesses it, runs the baseline, and trains the CNN.
4. Run the final test-evaluation cell only after training choices are fixed. The test set is for final reporting, not tuning.

For a local environment, install a PyTorch build suitable for your machine first, then run:

    pip install -r requirements.txt
    python -m src.prepare_data --dataset-root /path/to/PKLot --output-dir data/processed --seed 42 --max-crops 10000
    python -m src.baseline --manifest data/processed/manifest.csv
    python -m src.train --manifest data/processed/manifest.csv --epochs 20 --batch-size 64 --image-size 128 --seed 42
    python -m src.evaluate --manifest data/processed/manifest.csv --checkpoint outputs/models/best.pt

## Reproducibility and artifacts

- Default seed: 42
- Total dataset cap: 10,000 parking-space crops
- Target split: approximately 70% train / 15% validation / 15% test, grouped by capture date
- Input size: 128 × 128 (unchanged)
- Training and evaluation use every sample in their respective manifest split
- Optimizer: AdamW
- Loss: weighted cross-entropy (weights computed from the training split only)
- Regularization: augmentation, dropout, weight decay, and early stopping
- Model selection: validation loss only
- Test evaluation: separate final script

Training outputs, checkpoints, generated crops, and datasets are intentionally excluded from Git. The notebook and scripts save them in the Colab runtime. Record the actual runtime, hyperparameters, split counts, and metrics in the final report.

## Scope and limitation

This is a single parking-space crop classifier, not a complete object detector that discovers parking spaces in an arbitrary full-scene photograph. The demo expects one already-cropped parking-space image. The cross-site test may be substantially harder than a random image split; report that result honestly.
