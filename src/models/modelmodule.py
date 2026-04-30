from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import numpy as np
import warnings
import os
from PIL import Image

from src.utils.metrics import l2_relative_error
from src.utils.ntk import analyze_model_ntk


class INRTraining(LightningModule):
    """A Lightning Module for implicit neural representation training.
    """

    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: Any = None,
        scheduler: Any = None,
        criterion: Any = None,
        compile: bool = False,
        # NTK analysis parameters
        ntk_analysis: bool = False,
        ntk_frequency: int = 10,
        ntk_top_k: int = 10,
        ntk_normalize: str = "trace",
        # Checkpoint parameters
        checkpoint_epochs: list = None,
        ablation_noise: bool = False,
        noise_type: str = "gaussian",
        noise_level: float = 0.01,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # NTK analysis setup
        self.ntk_analysis = ntk_analysis
        self.ntk_frequency = ntk_frequency
        self.ntk_top_k = ntk_top_k
        self.ntk_normalize = ntk_normalize
        
        # Checkpoint setup
        self.checkpoint_epochs = checkpoint_epochs if checkpoint_epochs is not None else []
        
        # Noise setup
        self.ablation_noise = ablation_noise
        self.noise_type = noise_type
        self.noise_level = noise_level

        # metrics 
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()
        
        # NTK metrics
        if self.ntk_analysis:
            self.ntk_effective_rank = MeanMetric()
            self.ntk_condition_number = MeanMetric()
            self.ntk_spectrum_decay = MeanMetric()
            self.ntk_eigenvalue_metrics = {}

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

    def _get_ntk_inputs(self) -> torch.Tensor:
        """Get inputs for NTK analysis from DataModule's fixed coordinate subset.
        
        Returns:
            NTK input coordinates on CPU
        """
        # Get fixed coordinates from DataModule
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError(
                "Cannot get NTK inputs: trainer or datamodule not available. "
                "Ensure NTK analysis is run during training."
            )
        
        coords = self.trainer.datamodule.get_ntk_coords()
        if coords is None:
            raise RuntimeError(
                "DataModule has not initialized NTK coordinates. "
                "Ensure DataModule.setup() has been called."
            )
        
        return coords

    def _setup_ntk_analysis(self) -> None:
        """Setup NTK analysis components."""
        try:
            # Store NTK results for analysis
            self.ntk_results_history = []
            
            # Verify DataModule provides NTK coords
            if self.trainer.datamodule is None:
                raise RuntimeError("DataModule not available for NTK analysis")
            
            coords = self.trainer.datamodule.get_ntk_coords()
            if coords is None:
                raise RuntimeError("DataModule has not initialized NTK coordinates")
            
            print(f"✓ NTK analysis initialized: top-{self.ntk_top_k} eigenvalues logged, "
                  f"full spectrum saved to disk, frequency every {self.ntk_frequency} epochs, "
                  f"using {len(coords)} fixed coordinates from DataModule")
                  
        except Exception as e:
            warnings.warn(f"Failed to setup NTK analysis: {e}. Disabling NTK analysis.")
            self.ntk_analysis = False

    def _perform_ntk_analysis(self) -> Optional[Dict[str, float]]:
        """Perform NTK analysis on the current model state."""
        if not self.ntk_analysis:
            return None
            
        try:
            # Get inputs for NTK analysis (from DataModule's fixed coords)
            inputs = self._get_ntk_inputs()
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Perform NTK analysis (model automatically set to eval inside)
            result = analyze_model_ntk(
                self.model,
                inputs,
                normalize=self.ntk_normalize,
                top_k=self.ntk_top_k,
                return_all_eigenvalues=True
            )
            
            # Extract key metrics
            metrics = {
                'effective_rank': float(result.effective_rank),
                'condition_number': float(result.condition_number),
                'spectrum_decay': float(result.spectrum_decay),
                'trace': float(result.trace),
                'total_eigenvalues': len(result.eigenvalues),
            }
            
            # Add ALL eigenvalues to stored metrics (for disk saving)
            for i, eigenval in enumerate(result.eigenvalues):
                metrics[f'eigenvalue_{i+1}'] = float(eigenval)
            
            # Store for history tracking (full spectrum saved to disk)
            metrics['epoch'] = self.current_epoch
            self.ntk_results_history.append(metrics)
            
            # Log summary metrics
            self.ntk_effective_rank(metrics['effective_rank'])
            self.ntk_condition_number(metrics['condition_number'])
            self.ntk_spectrum_decay(metrics['spectrum_decay'])
            
            # Log only TOP-K eigenvalues to avoid metric explosion
            for i in range(min(self.ntk_top_k, len(result.eigenvalues))):
                eigenval = float(result.eigenvalues[i])
                metric_name = f'ntk_eigenvalue_{i+1}'
                
                # Create metric if not exists
                if metric_name not in self.ntk_eigenvalue_metrics:
                    self.ntk_eigenvalue_metrics[metric_name] = MeanMetric()
                    # Register the metric with the module for proper logging
                    setattr(self, metric_name, self.ntk_eigenvalue_metrics[metric_name])
                
                self.ntk_eigenvalue_metrics[metric_name](eigenval)
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"NTK analysis failed at epoch {self.current_epoch}: {e}")
            return None

    def _add_output_noise(self, gt, noise_type, noise_level):
        with torch.no_grad():
            x = gt.detach()
            if noise_type == "gaussian":
                noise = torch.randn_like(x) * noise_level
                x = x + noise
            elif noise_type == "uniform":
                noise = torch.rand_like(x) * noise_level
                x = x + noise
            else:
                raise ValueError(f"Unknown noise kind: {noise_type}")
            return x.clamp(0.0, 1.0)

    def on_fit_start(self) -> None:
        if self.ntk_analysis:
            self._setup_ntk_analysis()

    def _save_checkpoint_at_epoch(self) -> None:
        """Save model checkpoint at specific epoch."""
        try:
            output_dir = self.trainer.log_dir if self.trainer.log_dir is not None else "."
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{self.current_epoch}.ckpt")
            
            # Save the checkpoint using trainer
            self.trainer.save_checkpoint(checkpoint_path)
            print(f"✓ Checkpoint saved at epoch {self.current_epoch}: {checkpoint_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint at epoch {self.current_epoch}: {e}")

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step (forward pass + loss computation).
        
        Args:
            batch: A tuple of (coordinates, ground_truth).
            
        Returns:
            Tuple of (loss, predictions, ground_truth)
        """
        coords, gt = batch
        pred = self.model(coords)
        if self.ablation_noise:
            gt_noisy = self._add_output_noise(gt, self.noise_type, self.noise_level)
        else:
            gt_noisy = gt
        loss = self.criterion(pred, gt_noisy)
        return loss, pred, gt

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, pred, gt = self.model_step(batch)
        
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.train_loss(loss)
        self.train_rel_error(rel_error)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rel_error", self.train_rel_error, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

        if self.ntk_analysis and (self.current_epoch % self.ntk_frequency == 0):
            _ = self._perform_ntk_analysis()
            
        # Save checkpoint at predefined epochs
        if self.current_epoch in self.checkpoint_epochs:
            self._save_checkpoint_at_epoch()
            
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, pred, gt = self.model_step(batch)

        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.test_loss(loss)
        
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_ground_truth.append(gt.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Evaluate the model on the full data grid if supported."""
        preds = torch.cat(self.test_predictions)
        ground_truths = torch.cat(self.test_ground_truth)
        rel_error = l2_relative_error(preds.flatten(), ground_truths.flatten())
        self.save_data(preds.detach().cpu().numpy(), filename="test_preds")
        self.save_data(rel_error, filename="test_rel_error")
        
        # Save prediction as PNG if using ImageDataModule
        if hasattr(self.trainer.datamodule, 'get_image_shape'):
            self.save_prediction_as_png(preds, filename="prediction.png")
            # Also save ground truth for comparison
            self.save_prediction_as_png(ground_truths, filename="ground_truth.png")
        
        if self.ntk_analysis:
            self.save_ntk_results()
            
    def setup(self, stage: str) -> None:
        """Called at the beginning of fit and test."""
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            A dictionary containing the optimizer and scheduler configuration.
        """
        if hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.parameters())
        else:
            # Default optimizer if none is provided
            warnings.warn("No optimizer specified, using AdamW with default settings.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            lr_scheduler_config = {"scheduler": scheduler}

            # For ReduceLROnPlateau, we need to specify the metric to monitor
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "train/loss"

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
            
        return {"optimizer": optimizer}

    def save_data(self, data: np.ndarray, filename: str = "predictions.npy") -> None:
        """Save data to a file."""
        output_dir = self.trainer.log_dir
        np.save(f"{output_dir}/{filename}", data)
    
    def save_prediction_as_png(
        self, 
        predictions: torch.Tensor, 
        filename: str = "prediction.png"
    ) -> None:
        """Save predictions as a PNG image.
        
        This method reconstructs the image from flattened predictions and saves it.
        It automatically handles normalization inversion based on the DataModule settings.
        
        Args:
            predictions: Flattened predictions tensor of shape (H*W, C)
            filename: Output filename
        """
        try:
            datamodule = self.trainer.datamodule
            
            # Check if datamodule has image shape info (ImageDataModule)
            if not hasattr(datamodule, 'get_image_shape'):
                warnings.warn(
                    "DataModule does not support image shape retrieval. "
                    "PNG saving is only supported for ImageDataModule."
                )
                return
            
            H, W, C = datamodule.get_image_shape()
            
            # Reshape predictions to image format
            img = predictions.view(H, W, C).cpu().numpy()
            
            # Invert zero_mean normalization if it was applied: [-1,1] -> [0,1]
            if hasattr(datamodule.hparams, 'zero_mean') and datamodule.hparams.zero_mean:
                img = img * 0.5 + 0.5
            
            # Clip to valid range and convert to uint8
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
            
            # Handle single-channel images
            if C == 1:
                img = img.squeeze(-1)
            
            # Save using PIL
            output_dir = self.trainer.log_dir
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(img).save(output_path)
            print(f"✓ Prediction saved as PNG: {output_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save prediction as PNG: {e}")
        
    def save_ntk_results(self, filename: str = "ntk_analysis.npy") -> None:
        """Save NTK analysis results to file."""
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis and hasattr(self, 'ntk_results_history'):
            output_dir = self.trainer.log_dir
            if output_dir is not None:
                np.save(os.path.join(output_dir, filename), self.ntk_results_history)
                print(f"✓ NTK results saved to {output_dir}/{filename}")
            else:
                np.save(filename, self.ntk_results_history)
                print(f"✓ NTK results saved to {filename}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""    
        info = {} 
        try:
            if hasattr(self.model, 'get_param_count'):
                trainable_params, total_params = self.model.get_param_count()
            else:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
            info['trainable_params'] = trainable_params
            info['total_params'] = total_params
        except Exception as e:
            warnings.warn(f"Could not get parameter count: {e}")
            info['trainable_params'] = 'N/A'
            info['total_params'] = 'N/A'
            
        return info
    

class DINRTraining(LightningModule):
    """A Lightning Module for dynamical implicit neural representation training.
    """

    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: Any = None,
        scheduler: Any = None,
        criterion: Any = None,
        compile: bool = False,
        # NTK analysis parameters
        ntk_analysis: bool = False,
        ntk_frequency: int = 10,
        ntk_top_k: int = 10,
        ntk_normalize: str = "trace",
        # Checkpoint parameters
        checkpoint_epochs: list = None,
        ablation_ot_loss: bool = False,
        ablation_noise: bool = False,
        noise_type: str = "gaussian",
        noise_level: float = 0.01,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # NTK analysis setup
        self.ntk_analysis = ntk_analysis
        self.ntk_frequency = ntk_frequency
        self.ntk_top_k = ntk_top_k
        self.ntk_normalize = ntk_normalize
        
        # Checkpoint setup
        self.checkpoint_epochs = checkpoint_epochs if checkpoint_epochs is not None else []

        # OT loss setup
        self.ablation_ot_loss = ablation_ot_loss

        # Noise setup
        self.ablation_noise = ablation_noise
        self.noise_type = noise_type
        self.noise_level = noise_level

        # metrics 
        self.train_loss = MeanMetric()
        self.train_data_loss = MeanMetric()
        self.train_ot_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_data_loss = MeanMetric()
        self.test_ot_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()
        
        # NTK metrics
        if self.ntk_analysis:
            self.ntk_effective_rank = MeanMetric()
            self.ntk_condition_number = MeanMetric()
            self.ntk_spectrum_decay = MeanMetric()
            self.ntk_eigenvalue_metrics = {}

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

    def _get_ntk_inputs(self) -> torch.Tensor:
        """Get inputs for NTK analysis from DataModule's fixed coordinate subset.
        
        Returns:
            NTK input coordinates on CPU
        """
        # Get fixed coordinates from DataModule
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError(
                "Cannot get NTK inputs: trainer or datamodule not available. "
                "Ensure NTK analysis is run during training."
            )
        
        coords = self.trainer.datamodule.get_ntk_coords()
        if coords is None:
            raise RuntimeError(
                "DataModule has not initialized NTK coordinates. "
                "Ensure DataModule.setup() has been called."
            )
        
        return coords

    def _setup_ntk_analysis(self) -> None:
        """Setup NTK analysis components."""
        try:
            # Store NTK results for analysis
            self.ntk_results_history = []
            
            # Verify DataModule provides NTK coords
            if self.trainer.datamodule is None:
                raise RuntimeError("DataModule not available for NTK analysis")
            
            coords = self.trainer.datamodule.get_ntk_coords()
            if coords is None:
                raise RuntimeError("DataModule has not initialized NTK coordinates")
            
            print(f"✓ NTK analysis initialized: top-{self.ntk_top_k} eigenvalues logged, "
                  f"full spectrum saved to disk, frequency every {self.ntk_frequency} epochs, "
                  f"using {len(coords)} fixed coordinates from DataModule")
                  
        except Exception as e:
            warnings.warn(f"Failed to setup NTK analysis: {e}. Disabling NTK analysis.")
            self.ntk_analysis = False

    def _perform_ntk_analysis(self) -> Optional[Dict[str, float]]:
        """Perform NTK analysis on the current model state."""
        if not self.ntk_analysis:
            return None
            
        try:
            # Get inputs for NTK analysis (from DataModule's fixed coords)
            inputs = self._get_ntk_inputs()
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Perform NTK analysis (model automatically set to eval inside)
            result = analyze_model_ntk(
                self.model,
                inputs,
                normalize=self.ntk_normalize,
                top_k=self.ntk_top_k,
                return_all_eigenvalues=True
            )
            
            # Extract key metrics
            metrics = {
                'effective_rank': float(result.effective_rank),
                'condition_number': float(result.condition_number),
                'spectrum_decay': float(result.spectrum_decay),
                'trace': float(result.trace),
                'total_eigenvalues': len(result.eigenvalues),
            }
            
            # Add ALL eigenvalues to stored metrics (for disk saving)
            for i, eigenval in enumerate(result.eigenvalues):
                metrics[f'eigenvalue_{i+1}'] = float(eigenval)
            
            # Store for history tracking (full spectrum saved to disk)
            metrics['epoch'] = self.current_epoch
            self.ntk_results_history.append(metrics)
            
            # Log summary metrics
            self.ntk_effective_rank(metrics['effective_rank'])
            self.ntk_condition_number(metrics['condition_number'])
            self.ntk_spectrum_decay(metrics['spectrum_decay'])
            
            # Log only TOP-K eigenvalues to avoid metric explosion
            for i in range(min(self.ntk_top_k, len(result.eigenvalues))):
                eigenval = float(result.eigenvalues[i])
                metric_name = f'ntk_eigenvalue_{i+1}'
                
                # Create metric if not exists
                if metric_name not in self.ntk_eigenvalue_metrics:
                    self.ntk_eigenvalue_metrics[metric_name] = MeanMetric()
                    # Register the metric with the module for proper logging
                    setattr(self, metric_name, self.ntk_eigenvalue_metrics[metric_name])
                
                self.ntk_eigenvalue_metrics[metric_name](eigenval)
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"NTK analysis failed at epoch {self.current_epoch}: {e}")
            return None

    def _add_output_noise(self, gt, noise_type, noise_level):
        with torch.no_grad():
            x = gt.detach()
            if noise_type == "gaussian":
                noise = torch.randn_like(x) * noise_level
                x = x + noise
            elif noise_type == "uniform":
                noise = torch.rand_like(x) * noise_level
                x = x + noise
            else:
                raise ValueError(f"Unknown noise kind: {noise_type}")
            return x.clamp(0.0, 1.0)

    def on_fit_start(self) -> None:
        if self.ntk_analysis:
            self._setup_ntk_analysis()

    def _save_checkpoint_at_epoch(self) -> None:
        """Save model checkpoint at specific epoch."""
        try:
            output_dir = self.trainer.log_dir if self.trainer.log_dir is not None else "."
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{self.current_epoch}.ckpt")
            
            # Save the checkpoint using trainer
            self.trainer.save_checkpoint(checkpoint_path)
            print(f"✓ Checkpoint saved at epoch {self.current_epoch}: {checkpoint_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint at epoch {self.current_epoch}: {e}")

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step (forward pass + loss computation).
        
        Args:
            batch: A tuple of (coordinates, ground_truth).
            
        Returns:
            Tuple of (loss, predictions, ground_truth)
        """
        coords, gt = batch
        pred, ot_loss = self.model(coords)

        if self.ablation_noise:
            gt_noisy = self._add_output_noise(gt, self.noise_type, self.noise_level)
        else:
            gt_noisy = gt

        data_loss = self.criterion(pred, gt_noisy)
        if self.ablation_ot_loss:
            loss = data_loss
        else:
            loss = data_loss + ot_loss
        return data_loss, ot_loss, loss, pred, gt

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        data_loss, ot_loss, loss, pred, gt = self.model_step(batch)
        
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.train_data_loss(data_loss)
        self.train_ot_loss(ot_loss)
        self.train_loss(loss)
        self.train_rel_error(rel_error)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", self.train_data_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ot_loss", self.train_ot_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rel_error", self.train_rel_error, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

        if self.ntk_analysis and (self.current_epoch % self.ntk_frequency == 0):
            _ = self._perform_ntk_analysis()

        # Save checkpoint at predefined epochs
        if self.current_epoch in self.checkpoint_epochs:
            self._save_checkpoint_at_epoch()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        data_loss, ot_loss, loss, pred, gt = self.model_step(batch)

        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.test_loss(loss)
        self.test_data_loss(data_loss)
        self.test_ot_loss(ot_loss)
        
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_ground_truth.append(gt.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Evaluate the model on the full data grid if supported."""
        preds = torch.cat(self.test_predictions)
        ground_truths = torch.cat(self.test_ground_truth)
        rel_error = l2_relative_error(preds.flatten(), ground_truths.flatten())
        self.save_data(preds.detach().cpu().numpy(), filename="test_preds")
        self.save_data(rel_error, filename="test_rel_error")
        
        # Save prediction as PNG if using ImageDataModule
        if hasattr(self.trainer.datamodule, 'get_image_shape'):
            self.save_prediction_as_png(preds, filename="prediction.png")
            self.save_prediction_as_png(ground_truths, filename="ground_truth.png")
        
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis:
            self.save_ntk_results()
            
    def setup(self, stage: str) -> None:
        """Called at the beginning of fit and test."""
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            A dictionary containing the optimizer and scheduler configuration.
        """
        if hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.parameters())
        else:
            # Default optimizer if none is provided
            warnings.warn("No optimizer specified, using AdamW with default settings.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            lr_scheduler_config = {"scheduler": scheduler}

            # For ReduceLROnPlateau, we need to specify the metric to monitor
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "train/loss"

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
            
        return {"optimizer": optimizer}

    def save_data(self, data: np.ndarray, filename: str = "predictions.npy") -> None:
        """Save data to a file."""
        output_dir = self.trainer.log_dir
        np.save(f"{output_dir}/{filename}", data)
    
    def save_prediction_as_png(
        self, 
        predictions: torch.Tensor, 
        filename: str = "prediction.png"
    ) -> None:
        """Save predictions as a PNG image.
        
        This method reconstructs the image from flattened predictions and saves it.
        It automatically handles normalization inversion based on the DataModule settings.
        
        Args:
            predictions: Flattened predictions tensor of shape (H*W, C)
            filename: Output filename
        """
        try:
            datamodule = self.trainer.datamodule
            
            # Check if datamodule has image shape info (ImageDataModule)
            if not hasattr(datamodule, 'get_image_shape'):
                warnings.warn(
                    "DataModule does not support image shape retrieval. "
                    "PNG saving is only supported for ImageDataModule."
                )
                return
            
            H, W, C = datamodule.get_image_shape()
            
            # Reshape predictions to image format
            img = predictions.view(H, W, C).cpu().numpy()
            
            # Invert zero_mean normalization if it was applied: [-1,1] -> [0,1]
            if hasattr(datamodule.hparams, 'zero_mean') and datamodule.hparams.zero_mean:
                img = img * 0.5 + 0.5
            
            # Clip to valid range and convert to uint8
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
            
            # Handle single-channel images
            if C == 1:
                img = img.squeeze(-1)
            
            # Save using PIL
            output_dir = self.trainer.log_dir
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(img).save(output_path)
            print(f"✓ Prediction saved as PNG: {output_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save prediction as PNG: {e}")
        
    def save_ntk_results(self, filename: str = "ntk_analysis.npy") -> None:
        """Save NTK analysis results to file."""
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis and hasattr(self, 'ntk_results_history'):
            output_dir = self.trainer.log_dir
            if output_dir is not None:
                np.save(os.path.join(output_dir, filename), self.ntk_results_history)
                print(f"✓ NTK results saved to {output_dir}/{filename}")
            else:
                np.save(filename, self.ntk_results_history)
                print(f"✓ NTK results saved to {filename}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""    
        info = {} 
        try:
            if hasattr(self.model, 'get_param_count'):
                trainable_params, total_params = self.model.get_param_count()
            else:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
            info['trainable_params'] = trainable_params
            info['total_params'] = total_params
        except Exception as e:
            warnings.warn(f"Could not get parameter count: {e}")
            info['trainable_params'] = 'N/A'
            info['total_params'] = 'N/A'
            
        return info
