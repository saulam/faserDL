# MAE-ViT for FASERCal Neutrino Data

This repository provides an implementation of a Masked Autoencoder (MAE) Vision Transformer (ViT) integrated with a MinkowskiEngine-based sparse convolutional patcher, tailored for FASERCal neutrino detector data. It supports a two-stage learning pipeline:

* **Stage 1**: Masked pre-training.
* **Stage 2**: Muti-task fine-tuning.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)

  * [Scripts](#scripts)
  * [Command-line Arguments](#command-line-arguments)
* [Requirements](#requirements)
* [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/saulam/faserDL.git
   cd faserDL
   ```
2. (Optional) Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Scripts

* **Pretraining**:

  ```bash
  ./pretrain.sh
  ```

  Runs default MAE pretraining (stage1).

* **Finetuning**:

  ```bash
  ./finetune.sh
  ```

  Runs default fine-tuning (stage2) on pretrained checkpoints.

You can override any settings via command-line arguments (see below).

### Command-line Arguments

| Group                    | Arguments                                                                                                                                                           | Type & Default                                                                                                                                    | Description                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Mode**                 | `--train` / `--test`                                                                                                                                                | flag (`train=True` by default, `--test` sets `train=False`)                                                                                       | Switch between training and testing modes                                                                       |
| **Stage**                | `--stage1` / `--stage2`                                                                                                                                             | flag (`stage1=True` by default, `--stage2` sets `stage1=False`)                                                                                   | Select pipeline stage: pre-training (stage1) or fine-tuning (stage2)                                    |
| **Preprocessing**        | `--preprocessing_input`, `--preprocessing_output`                                                                                                                   | str (`"log"` / `"sqrt"`)                                                                                                                          | Input and output data transforms                                                                                 |
| **Standardisation**      | `--standardize_input`, `--standardize_output`                                                                                                                       | str (`"z-score"` / `"uni-var"` / `"norm"` / `None`)                                                                                               | Standardise input and output data                                                                                |
| **Augmentations**        | `--augmentations_enabled` / `--augmentations_disabled`                                                                                                              | flag (enabled by default)                                                                                                                         | Toggle data augmentations                                                                                        |
| **Regularisation / Mix** | `--label_smoothing`<br>`--mixup_alpha`                                                                                                                              | float (0.0)<br>float (0.0)                                                                                                                        | Label smoothing factor; mixup interpolation alpha                                                                |
| **Data & Model**         | `-d`, `--dataset_path`<br>`--extra_dataset_path`<br>`--metadata_path`<br>`--model`<br>`--mask_ratio`<br>`--eps`                                                      | str (required)<br>str (optional)<br>str (required)<br>choice (`base` / `large` / `huge`)<br>float (0.75)<br>float (1e-12)                            | Primary dataset directory; additional dataset to append; metadata file; model size; masking ratio; small epsilon |
| **Training**             | `-b`, `--batch_size`<br>`-e`, `--epochs`<br>`-w`, `--num_workers`<br>`-ag`, `--accum_grad_batches`<br>`--layer_decay`                                              | int (2)<br>int (50)<br>int (16)<br>int (1)<br>float (0.9)                                                                                          | Batch size; epochs; loader workers; gradient accumulation steps; layer-wise learning rate decay                   |
| **Schedulers**           | `-ws`, `--warmup_steps`<br>`--cosine_annealing_steps`                                                                                                               | int (0)<br>int (0)                                                                                                                                | Number of warmup steps for LR; number of cosine annealing steps                                                  |
| **Optimisation**         | `--lr`<br>`--blr`<br>`-wd`, `--weight_decay`<br>`-b1`, `--beta1`<br>`-b2`, `--beta2`<br>`--ema_decay`<br>`--head_init`<br>`--dropout`<br>`--drop_path_rate`          | float (None)<br>float (None)<br>float (0.05)<br>float (0.9)<br>float (0.999)<br>float (0.9999)<br>float (0.001)<br>float (0.0)<br>float (0.0)         | Learning rates; base LR; weight decay; AdamW betas; EMA decay; head init value; dropout; stochastic depth rate     |
| **Logging & Checkpoint** | `--save_dir`<br>`--name`<br>`--log_every_n_steps`<br>`--early_stop_patience`<br>`--save_top_k`<br>`--checkpoint_path`<br>`--checkpoint_name`<br>`--load_checkpoint` | str<br>str<br>int (50)<br>int (0)<br>int (1)<br>str<br>str<br>str                                                                                   | Directories/names for logs and checkpoints; logging freq; early stopping patience; how many top checkpoints to keep; checkpoint loading |
| **Hardware**             | `--gpus`                                                                                                                                                            | list of ints (e.g., `[0]`, parsed from strings)                                                                                                  | GPU device IDs; multiple IDs enable parallel/multi-GPU training                                                  |


## Requirements

```text
matplotlib==3.10.3
MinkowskiEngine==0.5.4
numpy==2.3.2
packaging==25.0
pytorch_lightning==2.0.0
ROOT==0.0.1
scikit_learn==1.7.1
scipy==1.16.0
timm==0.6.13
torch==2.0.0
torch_ema==0.3
tqdm==4.67.1
```

Ensure you have Python 3.10.5 and CUDA 11.8 installed to match the above dependencies.

## License

This project is licensed under the [MIT License](LICENSE).

