import datetime
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import trim_mean
from dataset.feature_names import SUBSETS_MASKS
from src.consts import *
from paths import ROOT_RUNS_DIR
from visuals.eval_plots import create_box_plot_comparison
import torch


def get_info_dictionary(stats_train, stats_val, model_type):
    keys = ["mean", "std", "max", "min"]
    categories = [f"train_{model_type}", f"val_{model_type}"]

    # Combine input tuples and their categories
    data = {
        categories[0]: stats_train,
        categories[1]: stats_val,
    }

    # Create the dictionary with formatted keys and corresponding values
    stats_dict = {}
    for category, values in data.items():
        for key, value in zip(keys, values):
            stats_dict[f"loss_{key}_{category}"] = value

    return stats_dict


class Trainer:
    """
    A trainer class to handle training and validation epochs for a Hawkes process model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        logging,
        cfg,
        device,
        run_dir=None,
        patience=4,
    ):
        """
        Initialize the trainer with the required components.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (callable): The loss function.
                Should accept (inputs, **kwargs_outputs).
            run_dir (str or Path): Directory where model weights will be saved.
        """
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logging = logging
        self.cfg = cfg

        self.loss_tracker_global = {
            "train": {self.cfg.model_type: []},
            "val": {self.cfg.model_type: []},
        }

        lr = self.optimizer.param_groups[0]["lr"]

        if run_dir is None:
            run_dir = ROOT_RUNS_DIR / (
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_lr_{lr}_"
                + str(random.randint(1, 1000))
            )

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.valid_likelihood = None
        self.test_likelihood = None
        self.last_epoch = None
        self.patience = patience
        self.delta = 0.001

        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

        self.lr_rescaling = cfg.lr_rescaling
        self.compute_probability = False

    #################
    # FORWARD PASS  #
    #################

    def forward_pass(
        self,
        model,
        input_features_stpp,
        input_features_dur,
        history_points,
        current_point,
        current_dur,
        radius=None,
        reduce="mean",
    ):

        (
            self.logging.info(f"Model is on training mode: {model.training}")
            if DEBUG
            else None
        )
        input_features_stpp = input_features_stpp.to(self.device)
        input_features_dur = input_features_dur.to(self.device)
        current_dur = current_dur.to(self.device)
        current_point = current_point.to(self.device)
        history_points = history_points.to(self.device)
        current_dur = current_dur.to(self.device)

        neural_pars = model(input_features_stpp, input_features_dur, current_dur)

        mu, alpha, beta, sigma, means, dur_rate = neural_pars

        kwargs_outputs = {
            "baseline_kwargs": {"mu": mu},
            "temporal_kwargs": {"alpha": alpha, "beta": beta},
            "spatial_kwargs": {"mean": means, "sigma": sigma},
        }

        if self.cfg.model_type == "saccade":

            loss, probability = self.criterion(
                current_point,
                history_points,
                **kwargs_outputs,
                compute_probability=self.compute_probability,
                radius=radius,
                reduce=reduce,
            )

        elif self.cfg.model_type == "duration":
            loss = self.criterion(
                current_dur[:, 0].unsqueeze(-1), dur_rate, reduce=reduce
            )

        if alpha is not None and sigma is not None:
            self.curr_alpha_avg = alpha.mean().item()
            self.curr_sigma = sigma.item()

        if DEBUG and self.model.training:
            with torch.no_grad():

                if self.cfg.model_type == "saccade":
                    mask = (beta == 0) * (alpha == 0) * (alpha == beta) != True
                    self.logging.info(
                        f" Alpha: {alpha.sum().item() / mask.sum()}, Beta: {beta.sum().item() / mask.sum()}, Sigma: {sigma.item()}"
                    )

                if self.cfg.model_type == "duration":

                    self.logging.info(
                        f"Dur  Mean: {dur_rate[0].mean().item()}, Variance: {dur_rate[1].mean().item()}"
                    )
        if self.cfg.model_type == "saccade":
            return loss, loss.detach(), probability
        elif self.cfg.model_type == "duration":
            return loss, loss.detach(), None

    #################
    # TRAINING LOOP #
    #################

    def train(self, train_loader, val_loader, epochs):
        """
        The main loop that runs ltiple epochs of training and validation.

        Args:
            train_dataset: Dataset used for training.
            val_dataset: Dataset used for validation.
            epochs (int): Number of epochs to train.

        Returns:
            self.model (torch.nn.Module): Trained model.
        """
        for epoch in range(1, epochs + 1):

            if self.lr_rescaling < 1 and epoch != 1:
                self.optimizer.param_groups[0]["lr"] = (
                    self.optimizer.param_groups[0]["lr"] * self.lr_rescaling
                )
                self.logging.info(
                    f"Rescaling Learning Rate of {self.lr_rescaling}, NEW LR: {self.optimizer.param_groups[0]['lr']}"
                )

            stats_train = self.train_epoch(train_loader, epoch)
            self.loss_tracker_global["train"][self.cfg.model_type].append(stats_train)
            self.logging.info(f"Epoch {epoch} - Training Loss: {stats_train[0]:.4f}")

            stats_val = self.validate_epoch(val_loader)
            self.logging.info(f"Epoch {epoch} - Validation Loss: {stats_val[0]:.4f}")
            self.loss_tracker_global["val"][self.cfg.model_type].append(stats_val)

            if self.check_early_stopping(stats_val[0], epoch=epoch):
                self.logging.info("Early stopping triggered. Training halted.")
                break

            if self.cfg.model_type == "saccade" and (
                self.curr_alpha_avg <= 0.01 or self.curr_sigma <= 1e-9
            ):
                self.logging.info("Alpha is zero, initializing parameters")
                self.model.initialize_parameters()
                self.model.to(self.device)
                learning_rate = self.optimizer.param_groups[0]["lr"]
                weight_decay = self.optimizer.param_groups[0]["weight_decay"]
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

    #################
    # TRAINING STEP #
    #################

    def train_epoch(self, loader, epoch=0):
        """
        Perform one training epoch over the given dataset.

        Args:
            dataset: An object with `items` attribute
                     and a __getitem__ that returns (inputs, outputs).
            epoch (int): Current epoch index, used for logging.

        Returns:
            float: The average training loss over the epoch.
        """
        self.model.train()

        loss_tracker = {self.cfg.model_type: []}

        for iteration, (
            input_features_stpp,
            input_features_dur,
            history_points,
            current_point,
            current_dur,
            _,
            _,
        ) in enumerate(loader):
            self.optimizer.zero_grad()

            loss, loss_detached, _ = self.forward_pass(
                self.model,
                input_features_stpp=input_features_stpp,
                input_features_dur=input_features_dur,
                history_points=history_points,
                current_dur=current_dur,
                current_point=current_point,
            )
            loss.backward()

            #######################
            # GRADIENT CLIPPING   #
            #######################

            if self.cfg.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            loss_tracker[self.cfg.model_type].append(loss_detached.item())

            with torch.no_grad():
                self.get_model_diagnostics(
                    loss=loss_detached,
                    iteration=iteration,
                    epoch=epoch,
                    tot_iterations=len(loader),
                )

        loss_np = np.array(loss_tracker[self.cfg.model_type])
        loss_stats = (
            loss_np.mean(),
            trim_mean(loss_np, proportiontocut=0.1),
            loss_np.std(),
            loss_np.max(),
            loss_np.min(),
        )

        return loss_stats

    ###################
    # VALIDATION STEP #
    ###################

    def validate_epoch(self, loader):
        """
        Perform one validation epoch over the given dataset.

        Args:
            dataset: An object with `items` attribute
                     and a __getitem__ that returns (inputs, outputs).

        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        loss_tracker = {self.cfg.model_type: []}

        with torch.no_grad():
            for (
                input_features_stpp,
                input_features_dur,
                history_points,
                current_point,
                current_dur,
                boxes,
                fixations_cond,
            ) in loader:

                loss, loss_detached, _ = self.forward_pass(
                    model=self.model,
                    input_features_stpp=input_features_stpp,
                    input_features_dur=input_features_dur,
                    history_points=history_points,
                    current_point=current_point,
                    current_dur=current_dur,
                )

                loss_tracker[self.cfg.model_type].append(loss_detached.item())

        loss_np = np.array(loss_tracker[self.cfg.model_type])
        loss_stats = (
            loss_np.mean(),
            trim_mean(loss_np, proportiontocut=0.1),
            loss_np.std(),
            loss_np.max(),
            loss_np.min(),
        )

        return loss_stats

    ########################
    # EARLY STOPPING       #
    ########################
    def check_early_stopping(self, valid_loss, epoch):
        stop = False

        if valid_loss < self.best_loss - self.delta:
            self.best_loss = valid_loss
            self.best_epoch = epoch
            self.logging.info(
                f"New best loss for model found at epoch {epoch}, saving model..."
            )
            best_model_dir = self.run_dir / f"best_{self.cfg.model_type}_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), best_model_dir / "model_weights.pt")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.logging.info(f" Early stopping triggered at epoch {epoch}.")
                stop = True

        return stop

    ########################
    # GRADIENT DIAGNOSTICS #
    ########################

    def get_model_diagnostics(self, loss, iteration, epoch, tot_iterations):
        """
        self.logging.info or log diagnostic information about gradients and current loss.

        Args:
            loss (float): Current loss value.
            iteration (int): Index of current iteration in epoch.
            epoch (int): Current epoch index.
        """

        if DEBUG:
            total_grad_norm = 0
            layer_grad_norms = {}

            grads_list = [
                (name, param.grad)
                for name, param in self.model.named_parameters()
                if param.grad is not None
            ]

            # Accumulate the squared gradient norms
            for name, grad in grads_list:
                layer_name = name.split(".")[0]
                if layer_name not in layer_grad_norms:
                    layer_grad_norms[layer_name] = 0
                layer_grad_norms[layer_name] += torch.sum(grad**2).item()

            self.logging.info("===== Gradient Norms by Layer =====")
            # Summarize and log
            for layer_name, norm_squared in layer_grad_norms.items():
                grad_norm = torch.sqrt(torch.tensor(norm_squared))
                total_grad_norm += grad_norm
                self.logging.info(f"Layer Name: {layer_name}")
                self.logging.info(f"Gradient Norm: {grad_norm.item():.4f}")
            self.logging.info("==============================")

        if self.cfg.model_type == "saccade":
            self.logging.info(
                f"Iteration {iteration}/{tot_iterations}, Epoch: {epoch}, Loss STPP: {loss.item():.4f}"
            )
        elif self.cfg.model_type == "duration":
            self.logging.info(
                f"Iteration {iteration}/{tot_iterations}, Epoch: {epoch}, Loss DUR: {loss.item():.4f}"
            )
        if DEBUG:
            self.logging.info(f"Total Gradient Norm: {total_grad_norm.item():.4f}")

    #####################
    #     TESTING       #
    #####################

    def test_evaluation(
        self,
        run_dir,
        model,
        test_loader,
        radius=None,
        mode="test",
        compute_probability=True,
    ):
        """
        Perform one validation epoch over the given dataset.

        Args:
            dataset: An object with `items` attribute
                        and a __getitem__ that returns (inputs, outputs).

        Returns:
            float: The average validation loss.
        """
        model.eval()
        track_loss = []
        probability_tracker = []
        data_subset_indices = []

        self.compute_probability = compute_probability

        with torch.no_grad():
            for (
                input_features_stpp,
                input_features_dur,
                history_points,
                current_point,
                current_dur,
                boxes,
                fixations_cond,
            ) in test_loader:

                loss, loss_detached, probability = self.forward_pass(
                    model=model,
                    input_features_stpp=input_features_stpp,
                    input_features_dur=input_features_dur,
                    history_points=history_points,
                    current_point=current_point,
                    current_dur=current_dur,
                    radius=radius,
                    reduce="none",
                )
                track_loss += loss_detached.flatten().tolist()

                if compute_probability and self.cfg.model_type == "saccade":
                    probability_tracker = (
                        probability_tracker + (probability.squeeze(-1)).tolist()
                    )
                data_subset_indices.append(fixations_cond)

        fixations_cond = np.array(torch.concatenate(data_subset_indices, dim=0))
        loss_np = np.array(track_loss)

        loss_dir = self.run_dir / f"loss_results_{self.cfg.model_type}_{mode}"

        loss_dir.mkdir(parents=True, exist_ok=True)

        create_box_plot_comparison(
            loss_np,
            fixations_cond,
            list(SUBSETS_MASKS.keys()),
            filename=loss_dir / "box_plot.png",
        )
        np.save(loss_dir / f"{mode}_loss_{self.cfg.model_type}.npy", loss_np)

        # np.save(loss_dir / f"{mode}_probability.npy", np.array(probability_tracker))
        np.save(
            loss_dir / f"{mode}_subsets_flags_{self.cfg.model_type}.npy", fixations_cond
        )

        test_avg, test_std, test_avg_01, test_avg_005, test_avg_001, test_med = (
            loss_np.mean(),
            loss_np.std(),
            trim_mean(loss_np, proportiontocut=0.1),
            trim_mean(loss_np, proportiontocut=0.05),
            trim_mean(loss_np, proportiontocut=0.01),
            np.median(loss_np),
        )

        pd.DataFrame(
            {
                f"test_loss_avg_{self.cfg.model_type}": [test_avg],
                f"test_loss_std_{self.cfg.model_type}": [test_std],
                f"test_loss_avg_trimmed_0.1": [test_avg_01],
                f"test_loss_avg_trimmed_0.05": [test_avg_005],
                f"test_loss_avg_trimmed_0.01": [test_avg_001],
                f"test_loss_median": [test_med],
            }
        ).to_csv(run_dir / f"{mode}_loss_{self.cfg.model_type}.csv")

    def save_stats(self, run_dir):
        # Save aggregated metrics to CSV
        metrics_df = create_metrics_df(self.loss_tracker_global)
        metrics_df.to_csv(run_dir / "metrics.csv", index=False)
        try:
            save_train_val_plots(self.loss_tracker_global, run_dir)
        except Exception as e:
            self.logging.error(f"Error saving training/validation plots: {e}")

    def plot_gamma_params(self, run_dir: Path):
        """
        Plots the evolution of gamma parameters stored in self.model.track_gamma_values.

        Args:
            run_dir (Path): Directory where plots will be saved.
        """
        # Ensure run_dir exists
        run_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve the tracked values
        dic = self.model.track_gamma_values

        # Define colors and parameter names
        params = list(dic.keys())
        colors = ["blue", "orange", "green"]

        # Individual plots
        for i, param in enumerate(params):
            values = dic[param]
            plt.figure()
            plt.plot(range(len(values)), values, marker="o", linestyle="-")
            plt.title(f"Evolution of {param}")
            plt.xlabel("Step")
            plt.ylabel(param)
            plt.grid(True)
            file_path = run_dir / f"{param}.png"
            plt.savefig(file_path)
            plt.close()

        # Combined plot
        plt.figure()
        for i, param in enumerate(params):
            values = dic[param]
            plt.plot(range(len(values)), values, marker="o", linestyle="-", label=param)
        plt.title("Evolution of Gamma Parameters")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        combined_path = run_dir / "gamma_params_all.png"
        plt.savefig(combined_path)
        plt.close()

        print(f"Saved individual plots for {params} and combined plot to {run_dir}")


import matplotlib.pyplot as plt


def save_train_val_plots(metrics_dict, run_dir):
    """
    Generates and saves separate plots for training and validation loss curves
    (mean with shaded min-max) for all loss types in metrics_dict.

    Args:
        metrics_dict (dict): Dictionary structured as:
            {
                'train': {
                    'loss_type1': [(mean, trimmed, std, max, min), ...],
                    'loss_type2': [...],
                    ...
                },
                'val': {
                    'loss_type1': [(mean, trimmed, std, max, min), ...],
                    'loss_type2': [...],
                    ...
                }
            }
        run_dir (pathlib.Path): Directory where the plots will be saved.
    """
    # Determine number of epochs from a sample entry
    sample_mode = next(iter(metrics_dict))
    sample_loss_type = next(iter(metrics_dict[sample_mode]))
    n_epochs = len(metrics_dict[sample_mode][sample_loss_type])
    epochs = list(range(n_epochs))

    # Determine all loss types across train and val
    all_loss_types = sorted(
        set(metrics_dict.get("train", {}).keys() | metrics_dict.get("val", {}).keys())
    )

    # ---- Training Plot ----
    num_train_types = len(
        [lt for lt in all_loss_types if lt in metrics_dict.get("train", {})]
    )
    if num_train_types > 0:
        # Create figure with one subplot per loss type
        fig_train, axes_train = plt.subplots(
            num_train_types, 1, figsize=(8, 4 * num_train_types)
        )
        if num_train_types == 1:
            axes_train = [axes_train]

        idx = 0
        for loss_type in all_loss_types:
            if loss_type in metrics_dict.get("train", {}):
                ax = axes_train[idx]
                train_tuples = metrics_dict["train"][loss_type]
                train_mean = [tup[0] for tup in train_tuples]
                train_min = [tup[4] for tup in train_tuples]
                train_max = [tup[3] for tup in train_tuples]

                ax.plot(epochs, train_mean, label=f"Train {loss_type} Mean")
                ax.fill_between(
                    epochs,
                    train_min,
                    train_max,
                    alpha=0.2,
                    label=f"Train {loss_type} Min-Max",
                )

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Training Loss: {loss_type}")
                ax.legend()
                ax.grid(True)

                idx += 1

        plt.tight_layout()
        train_plot_path = run_dir / "plot_train.png"
        fig_train.savefig(train_plot_path)
        plt.close(fig_train)

    # ---- Validation Plot ----
    num_val_types = len(
        [lt for lt in all_loss_types if lt in metrics_dict.get("val", {})]
    )
    if num_val_types > 0:
        # Create figure with one subplot per loss type
        fig_val, axes_val = plt.subplots(
            num_val_types, 1, figsize=(8, 4 * num_val_types)
        )
        if num_val_types == 1:
            axes_val = [axes_val]

        idx = 0
        for loss_type in all_loss_types:
            if loss_type in metrics_dict.get("val", {}):
                ax = axes_val[idx]
                val_tuples = metrics_dict["val"][loss_type]
                val_mean = [tup[0] for tup in val_tuples]
                val_min = [tup[4] for tup in val_tuples]
                val_max = [tup[3] for tup in val_tuples]

                ax.plot(epochs, val_mean, label=f"Val {loss_type} Mean")
                ax.fill_between(
                    epochs,
                    val_min,
                    val_max,
                    alpha=0.2,
                    label=f"Val {loss_type} Min-Max",
                )

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Validation Loss: {loss_type}")
                ax.legend()
                ax.grid(True)

                idx += 1

        plt.tight_layout()
        val_plot_path = run_dir / "plot_val.png"
        fig_val.savefig(val_plot_path)
        plt.close(fig_val)


# Example usage (assuming metrics_dict and run_dir are defined):
# save_train_val_plots(metrics_dict, run_dir)


"""  
    def get_previous_point(history_points, current_point):
        curr_idx_hist = (
            ((history_points == current_point.unsqueeze(1)).prod(axis=2) == 1)
            .nonzero(as_tuple=False)[:, 1]
            .unsqueeze(-1)
        )
        zeros_column = torch.zeros(history_points.shape[0], 3)

        # Concatenate along the second dimension (dim=1)
        history_points_shifted = torch.cat(
            tensors=(zeros_column.unsqueeze(1), history_points), dim=1
        )
        indices = curr_idx_hist.unsqueeze(-1).expand(-1, -1, 3)
        return torch.gather(history_points_shifted, 1, indices)
 """

import pandas as pd
import numpy as np


def create_metrics_df(metrics_dict):
    """
    Given a dictionary with structure like:

        {
         'train': {
                    'model_type': [ (mean, mean_trimmed, std, max, min), ... ],
                   },
         'val': {
                    'model_type': [ (mean, mean_trimmed, std, max, min), ... ],
                   }
        }

    this function returns a DataFrame where each row corresponds to one epoch.
    The columns are named in the format "{Mode}_Loss_{model_type}_{Stat}", for example:
      "Train_Loss_saccade_Mean"

    It also adds the following columns:
      - "Current_epoch": the current epoch index.
      - "Best_Val_epoch": computed as the epoch index for which the average of all
         validation mean losses (over all loss types) is minimum.
      - "Best_Val_mean": computed as the epoch index where the validation mean is lowest.
    """

    # Determine the number of epochs from one of the metric lists.
    sample_mode = next(iter(metrics_dict))
    sample_loss_type = next(iter(metrics_dict[sample_mode]))
    n_epochs = len(metrics_dict[sample_mode][sample_loss_type])

    # List to store a dictionary for each epoch.
    rows = []
    for epoch in range(n_epochs):
        row = {}
        # Loop over each mode and loss type
        for mode in metrics_dict:
            for loss_type in metrics_dict[mode]:
                # Each entry is assumed to be a tuple of 4 np.float64 numbers:
                # (mean, std, max, min)
                tup = metrics_dict[mode][loss_type][epoch]
                col_mean = f"{mode.capitalize()}_Loss_{loss_type.upper()}_Mean"
                col_trimmed = f"{mode.capitalize()}_Loss_{loss_type.upper()}_Trimmed"
                col_std = f"{mode.capitalize()}_Loss_{loss_type.upper()}_Std"
                col_max = f"{mode.capitalize()}_Loss_{loss_type.upper()}_Max"
                col_min = f"{mode.capitalize()}_Loss_{loss_type.upper()}_Min"

                row[col_mean] = float(tup[0])
                row[col_trimmed] = float(tup[1])
                row[col_std] = float(tup[2])
                row[col_max] = float(tup[3])
                row[col_min] = float(tup[4])
        # Add the current epoch index.
        row["Current_epoch"] = epoch
        rows.append(row)

    # Create the DataFrame.
    df = pd.DataFrame(data=rows)

    # Compute Best_Val_epoch based on the average of all validation mean losses.
    val_mean_columns = [
        col
        for col in df.columns
        if col.startswith("Val_Loss_") and col.endswith("_Mean")
    ]
    if val_mean_columns:
        df["Val_mean_avg"] = df[val_mean_columns].mean(axis=1)
        best_val_epoch = int(df["Val_mean_avg"].idxmin())
        df["Best_Val_epoch"] = best_val_epoch
        df.drop(columns=["Val_mean_avg"], inplace=True)
    else:
        df["Best_Val_epoch"] = None

    return df
