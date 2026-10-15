# Fungal Segmentation in Microscopic Vine Wood Images

This project prepares the downloaded dataset for **binary semantic segmentation** on a separate training computer. The user's request is to train on a friend's laptop, so this preparation run does **not** install CUDA/PyTorch, download model weights, or start training.

## Dataset discovery (verified)

The original downloaded archive is `data/raw/An eye on the vine.zip`; it is preserved unchanged. Its contents were extracted to `data/extracted/An eye on the vine/`. The archive's own `Readme.md` states that it contains paired images and manually obtained binary masks, and describes the 427 synthetic and 881 real 256×256 thumbnails. It also contains 16 full-resolution microscope image/mask pairs for reference.

Actual archive inspection found:

- 1,324 decodable image files and 1,324 same-name mask pairs; no missing pairs, dimension mismatches, or decode failures.
- 1,308 thumbnail image/mask pairs for training preparation: 427 synthetic, 569 real in “With fungi”, and 312 real in “Without fungi”.
- 16 full-resolution microscope pairs, retained as reference-only and not mixed with thumbnail training samples.
- Thumbnail inputs are RGB TIFF at 256×256. The 16 reference images are RGB PNG at 1,392×1,040, 2,048×1,536, or 2,304×2,304.
- Real masks are predominantly black background/white fungus; 312 negative thumbnails and 11 “With fungi” thumbnails have empty masks. Four pixels in real masks use a rare extra color and are marked ignore (255), not guessed.
- Synthetic mask TIFFs are grayscale encoded as RGB and include antialiased intermediate gray values. A central, configurable threshold of 128 maps them to the documented binary target. This is a documented, reviewable preprocessing assumption, not an author-specified threshold.
- Exact duplicate detection finds 159 duplicate-image groups / 315 extra thumbnail records. Duplicate image hashes have identical associated mask contents. They are preserved, but each exact-image group is kept within one split to prevent leakage.

## Task/model compatibility

**Yes, YOLO can train on the `.tif` images, using the right YOLO task.** The source masks are binary semantic class maps, not individual-object polygons. Current Ultralytics documentation supports **YOLO26 semantic segmentation** with single-channel class-ID PNG masks, including `255` as ignore; it also lists TIFF/TIF among supported image formats. Use the `-sem` model/task (for example `yolo26n-sem.pt`), not the familiar `-seg` instance model, which expects per-instance polygons. The setup below builds the Ultralytics-compatible semantic dataset YAML and a duplicate-free image split. See the [YOLO semantic dataset format](https://docs.ultralytics.com/datasets/semantic/), [semantic task](https://docs.ultralytics.com/tasks/semantic/), and [prediction TIFF format list](https://docs.ultralytics.com/modes/predict/#image-and-video-formats).

## Layout

- `data/raw/`: downloaded archive; never modified by the scripts.
- `data/extracted/`: unmodified extraction of the archive.
- `data/processed/semantic_segmentation/images/` and `masks/`: byte-identical thumbnail images plus derived single-channel class-ID masks, organized by split.
- `data/processed/semantic_segmentation/dataset.yaml`: portable semantic dataset description for the original all-record split (not an instance-polygon YOLO-seg YAML).
- `data/processed/yolo_semantic_unique/`: duplicate-free materialization using Ultralytics semantic segmentation's `images/`, `masks/`, and `dataset.yaml` format; images remain TIFF and class maps are PNG.
- `data/splits/splits.json`: deterministic split manifest with source provenance and image hashes.
- `data/splits/train_unique.json`, `val_unique.json`, `test_unique.json`: deterministic manifests with one representative per exact image hash; original records are not deleted. Prefer unique manifests in the training/evaluation loader so repeated images do not overweight optimization or metrics.
- `configs/dataset.yaml`: single source of truth for dataset paths, classes, threshold, seed, groups, and split ratios.
- `reports/`: generated inventory, annotation/quality/preprocessing/validation reports and visual overlays.

Class IDs are `0=background`, `1=fungus`, `255=ignore`. Native image dimensions and pixel data are preserved; no resizing, image enhancement, color conversion, or lossy image recompression is applied. Only the source masks are converted to class-ID PNG files.

**Why can a processed mask look completely black?** Training masks store fungus as the integer class ID `1`, not display brightness `255`. In a normal 8-bit image viewer, `1/255` is almost black. The masks are not all empty: inspection found 985 thumbnail masks with at least one fungus pixel and 323 genuinely empty masks (312 documented negative images plus 11 empty masks in the “With fungi” folder). The visualization script scales class `1` to white for review only; it leaves the training mask values unchanged. Open the generated files in `reports/visualizations/` to inspect them.

## Environment

Use Python 3.10 or newer. In a fresh virtual environment, install the base data-preparation requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The base requirements deliberately do not install PyTorch, CUDA, Ultralytics, or model weights. The dedicated YOLO training setup is below. Ultralytics is distributed under AGPL-3.0 (with separate Enterprise licensing); review that license for your intended use.

## YOLO26 semantic training setup (friend's laptop)

These instructions use current Ultralytics **semantic segmentation**. Do not use `yolo26n-seg.pt` / `yolo segment train` for this dataset; those are instance segmentation. `yolo26n-sem.pt` is the small semantic baseline. Ultralytics downloads the pretrained checkpoint automatically on first load; the optional download script below triggers and verifies that load. No model was downloaded on this preparation machine.

### 1. Copy the project and prepared dataset

Transfer the repository **including** the Git-ignored generated folders `data/processed/yolo_semantic_unique/` and `data/splits/*_unique.json`. These are not committed by Git. The training laptop needs the processed unique dataset; keep the original archive if you need to regenerate it.

### 2. Create a clean environment and install GPU PyTorch

On Linux (or Windows PowerShell, changing the activation command accordingly), use Python 3.10 or 3.11. Install the PyTorch build matched to the laptop's NVIDIA driver using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) **before** installing Ultralytics. Do not copy a CUDA wheel command from another machine: CUDA wheel compatibility depends on the current PyTorch release and supported driver.

Linux example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Paste the exact CUDA-enabled torch/torchvision command generated at pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Continue with GPU training only when `torch.cuda.is_available()` prints `True` and the reported device is the intended RTX 3050. The training script fails instead of silently running on CPU when `--device 0` is requested and CUDA is missing.

### 3. Confirm the semantic dataset YAML

The generated [YOLO semantic dataset YAML](data/processed/yolo_semantic_unique/dataset.yaml) follows the official mask-folder schema:

- `path`: dataset root (`.` relative to the YAML file).
- `train`, `val`, `test`: the three image directories.
- `masks_dir: masks`: parallel directories containing one PNG mask per image, same stem.
- `names`: class ID 0 background and class ID 1 fungus.
- Mask value 255 means ignore; values 0 and 1 are the two classes.

The raw training images stay `.tif`; the derived label masks are `.png`. There is no polygon conversion. Before training, inspect a few pairs under `images/train` and `masks/train` and review the threshold note in `reports/annotation_analysis.md`.

If rebuilding on the friend’s laptop (with the extracted source tree also transferred), run the preparation sequence from the repository root:

```bash
python -m src.inspect_dataset
python -m src.analyze_annotations
python -m src.preprocess_images
python -m src.split_dataset
python -m src.experiment_preparation
python -m src.prepare_yolo_semantic
python -m src.validate_dataset
```

### 4. Download/verify the pretrained semantic checkpoint

This command causes Ultralytics to fetch `yolo26n-sem.pt` from its official model release if it is not already in the Ultralytics cache. Network access is needed on the first run; the checkpoint is not included in this repository.

```bash
python -m src.download_model --weights yolo26n-sem.pt
```

The same download happens automatically if the training script loads the model first. To use another supported official semantic checkpoint, check its name in the [Ultralytics YOLO26 semantic model docs](https://docs.ultralytics.com/tasks/semantic/#models) and pass its exact `-sem` name. Do not pass an instance `-seg` checkpoint.

### 5. Train the baseline

The baseline starts with unique training images, `imgsz=256` (the native thumbnail size), batch 4, up to 100 epochs, early-stop patience 15, fixed seed 42, AMP enabled, and no image/color/mosaic augmentation. This avoids altering fluorescence appearance before there is a baseline. On a smaller GPU, try `--batch 2`; if stable, later compare a separate run at a larger input size. The script refuses to overwrite an existing run name.

```bash
python -m src.train_yolo_semantic \
	--weights yolo26n-sem.pt \
	--device 0 \
	--imgsz 256 \
	--batch 4 \
	--epochs 100 \
	--patience 15 \
	--name yolo26n_sem_baseline
```

Run with an explicit `--device cpu` only if GPU training is unavailable. The script checks that the loaded checkpoint's task is semantic and trains from `data/processed/yolo_semantic_unique/dataset.yaml`; it saves Ultralytics logs and checkpoints under `experiments/yolo26n_sem_baseline/`, including `weights/best.pt` and `weights/last.pt`.

To resume a run, use Ultralytics’ resume API on `last.pt` (from the activated environment):

```bash
python -c "from ultralytics import YOLO; YOLO('experiments/yolo26n_sem_baseline/weights/last.pt').train(resume=True)"
```

### 6. Validate and evaluate once on the held-out test set

Training validation uses only the validation split. After choosing a model/run using validation results, evaluate the unique test images once with the project evaluator. It excludes ignore pixels and reports per-class IoU, Dice, precision, recall, and pixel accuracy, plus prediction masks and red overlays.

```bash
python -m src.evaluate_yolo_semantic \
	--weights experiments/yolo26n_sem_baseline/weights/best.pt \
	--device 0 \
	--imgsz 256
```

Results are written to `reports/evaluation_report.md`, `reports/evaluation_results.json`, and `predictions/test_evaluation/`. These are semantic pixel metrics; do not report instance mask mAP for this semantic ground truth.

### 7. Predict on a new TIFF or directory

The predictor accepts a single `.tif`/`.tiff` image or recursively scans a directory for supported image formats. It saves class-ID PNG masks and visualization overlays separately and does not alter input files.

```bash
python -m src.predict_yolo_semantic \
	--weights experiments/yolo26n_sem_baseline/weights/best.pt \
	--source /path/to/new/microscope_image.tif \
	--output predictions/new_images \
	--device 0 \
	--imgsz 256
```

For a directory, set `--source /path/to/image_folder`. Prediction mask values are 0 (background) and 1 (fungus); the visualization overlay paints predicted fungus red. Outputs preserve source image dimensions; the script errors rather than silently stretching a mismatched output.

### GPU and memory notes

An RTX 3050 is available in multiple VRAM configurations. Begin with batch 4 at 256 pixels and check `nvidia-smi`. If CUDA out-of-memory occurs, lower batch to 2 or 1 before considering input-size changes. Do not reduce image size below the original 256×256. To preserve thin structures, compare an independent 512-pixel run only if memory permits; upsampling cannot recover details absent from the source. Use validation metrics and overlays to choose, then evaluate the selected checkpoint on test once.

Official references: [Ultralytics semantic segmentation](https://docs.ultralytics.com/tasks/semantic/), [semantic dataset format](https://docs.ultralytics.com/datasets/semantic/), [training arguments](https://docs.ultralytics.com/modes/train/), [prediction and supported image formats](https://docs.ultralytics.com/modes/predict/), and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Prepare/rebuild data

Run from the repository root. Inspection/extraction reports are repeatable; preprocessing writes to separate generated directories and does not change the archive or extracted source files.

```bash
python -m src.inspect_dataset
python -m src.analyze_annotations
python -m src.preprocess_images
python -m src.split_dataset
python -m src.validate_dataset
python -m src.visualize_annotations
python -m src.analyze_mask_thresholds
python -m src.experiment_preparation
python -m src.prepare_yolo_semantic
```

The archive has already been extracted under `data/extracted/`. If it needs to be extracted again, use the platform's ZIP utility and extract into `data/extracted/` (never over `data/raw/`). Change dataset paths or the synthetic-mask threshold centrally in `configs/dataset.yaml`, then rerun all preparation steps.

Split by exact source-image SHA-256 groups, fixed seed 42, approximately 70/20/10 over 993 unique thumbnail images. This results in 694/199/100 unique images; because repeated source records are retained, the materialized record totals are 895/271/142 for train/validation/test. Source image and mask remain together, and exact duplicates do not cross splits. Do not use test for model selection.

## Review before training

Inspect `reports/annotation_analysis.md`, `reports/preprocessing_report.md`, `reports/validation_report.md`, and the images in `reports/visualizations/`. Especially review the synthetic grayscale-to-binary threshold and ignore-colored real-mask pixels. `data/processed/yolo_semantic_unique/dataset.yaml` is generated for Ultralytics' semantic PNG-mask loader. The older `data/processed/semantic_segmentation/dataset.yaml` remains a portable preparation manifest and contains repeated records; prefer the unique YOLO dataset for the documented training workflow.

`src.analyze_mask_thresholds` compares fixed-128 and Otsu binarization on the synthetic **mask annotations** only. It writes a report and visual comparison, but intentionally does not alter the training masks. Otsu is a threshold-selection method, not an edge detector or proof of better ground truth. Edge channels or boundary-aware losses are potential model experiments; compare them on validation before adopting them, and do not run edge filters on the labels to manufacture sharper ground truth.

`python -m src.experiment_preparation` reviews the 11 empty masks in the “With fungi” folder, verifies whether exact duplicate images have the same mask content, quantifies duplicate exposures, and writes unique-hash manifests for all three splits. It also creates `reports/visualizations/empty_with_fungi_mask_audit.png`. These manifests are for a future training/evaluation loader; the current project does not have a loader that consumes them yet. `src.paired_augmentation` provides tested, interpolation-free right-angle flips/rotations applied identically to image/mask pairs; it is not applied to files during preprocessing. `configs/experiments.yaml` records the controlled training comparisons to run once a semantic trainer and GPU environment are selected.

Validation currently covers readable files, paired dimensions, allowed class values, manifest/sample consistency, and exact-image-hash leakage. It passes. Near-duplicate detection and model-specific dataloader compatibility checks are not included.

## Training status and limitations

- Preprocessing and dataset preparation: completed and validated on this machine.
- YOLO26 semantic training/inference scripts and setup guide are ready, but the friend-laptop CUDA compatibility has not been inspected.
- Training, test-set model metrics, checkpoint, and predictions: not run/generated.
- No YOLO model checkpoint or training result is claimed. Use `-sem` semantic weights; instance `-seg` weights do not match these labels.

## Tests

Run the lightweight preprocessing tests (no dataset/GPU required):

```bash
python -m pytest -q
```

The downloaded dataset is distributed under CC BY-SA 4.0 as stated in its bundled README. Preserve that dataset attribution/license when redistributing it or derived materials. Ultralytics code and weights have their own AGPL-3.0/Enterprise licensing terms; review them separately.
