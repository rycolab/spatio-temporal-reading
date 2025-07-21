# A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior

This repository contains the code for the paper[&#34;A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior&#34;](https://arxiv.org/abs/2506.19999), to be presented at ACL 2025.

Below an animation of our hawkes process model evaluated on a reading sequence that was held out from the training and validation set:

![Reading session animation](image/held_out_sess.gif)

## ⚙️ Environment Setup

To run the experiments, we recommend setting up a Python environment with the required dependencies. You can do this using `conda`:

```bash

# Create a new environment with Python 3.12.7

conda create -n reading-models python=3.12.7 -y

conda activate reading-models


# Install dependencies

pip install -r requirements.txt
```

## 🧑‍💻 Example Usage

### 🌱 Single model run (local)

We recommend starting with the **model cards** defined in `scripts/model_cards.py`:

```bash
# List all predefined model variants from the paper, grouped in saccade and duration models, on filtered or full (raw) scanpaths
python scripts/launcher_local.py -h

# example: Train & evaluate the hawkes process with a spatial shift, reader specific effects, and word length as predictor on the raw scanpath dataset 
python scripts/launcher_local.py --rme-css-len-freq-raw

# example: Train & evaluate duration model with word suprisal and a convolved past spillover effects
python scripts/launcher_local.py --dur-rme-ws-raw

```

Each card exposes three boolean flags you can override via the `model_cards.py` script directly:

* `training=true|false` – perform parameter learning
* `testing=true|false`  – evaluate on held‑out split
* `subset=true|false`   – restrict to ~2 k events for quick smoke tests

More granular settings (learning rate, kernel size, etc.) live in the dataclass `config.py::RunConfig` and can be overridden via environment variables or direct YAML edits.

### ⚡ Hyper‑parameter tuning on HPC Cluster (SLURM)

Our experiments are executed on ETH Zürich’s Euler high-performance computing (HPC) cluster.  We define the search grid in `scripts/experiments.py`, then launch multi‑jobs:

```bash
python scripts/cluster_launcher_mjobs.py \
    --model <model_name> \
    --output-dir <output_directory> \
    --partition <partition_name> \
    --account <account_name> \
    --cpus <num_cpus> \
    --gpus <num_gpus> \
    --mem <memory_in_GB> \
    --time <time_limit>

```

### Selecting and Evaluating the Best Models

All experiments are saved under the `cluster_runs/` directory, which is organized a follows :

```bash
cluster_runs/
├── duration/
│   └── <model_name>_<timestamp>/
│       ├── [folder run 1]
│       ├── [folder run 2]
│       └── ... 
├── saccade/
│   └── <model_name>_<timestamp>/
│       ├── [folder run 1]
│       ├── [folder run 2]
│       └── ...
```

After running the experiments, we select the best model for each experiment folder based on its performance on the validation set.

To do this, run:

* For  **duration modeling** :
  ```bash
  python scripts/select_best_val_model.py --duration
  ```
* For  **saccade modeling** :
  ```bash
  python scripts/select_best_val_model.py --saccade
  ```

This script saves the best-performing model for each experiment in the `best_model/` directory.

---

### Running Test Evaluation

Once the best models are saved, evaluate them on the test set by running:

```bash
python scripts/run_test_eval_global.py --root-dir <path-to-dir>
```

This script:

* Loads every best-model checkpoint,
* Evaluates it on the test set, and
* For the **saccade modeling task**, creates an animation on a held-out reading session (reader 70, text 3)

  (see example animation above).

Citation

If you use this work, please cite it as:

```bibtex
@inproceedings{re2025spatiotemporal,
  title     = {A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior},
  author    = {Re, Francesco Ignazio and Opedal, Andreas and Manaiev, Glib and Giulianelli, Mario and Cotterell, Ryan},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  address   = {Vienna, Austria},
  month     = jul,
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.19999}
}
```
