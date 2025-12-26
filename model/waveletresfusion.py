import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from utils.dwt import DWTForward, DWTInverse

class GaussianResfusion_Restore(pl.LightningModule):
    """
    Core algorithm for Wavelet-based Heteroscedastic Diffusion.
    Implements the core noise injection and sampling logic.
    """

    def __init__(self, denoising_module, variance_scheduler, mode='residual', **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['denoising_module', 'variance_scheduler'])
        self.denoising_module = denoising_module
        self.var_scheduler = variance_scheduler
        self.mode = mode
        
        # Wavelet operators
        self.dwt = DWTForward()
        self.iwt = DWTInverse(in_channels=3)

        # Acceleration Point (T_acc) based on Equation 27
        self.T_acc = self.get_acc_point(self.var_scheduler.get_alphas_hat()).item()

    def get_acc_point(self, alphas_hat):
        """Calculate the acceleration jump point for efficient sampling"""
        return torch.abs(torch.sqrt(alphas_hat) - 0.5).argmin() + 1

    def training_step(self, batch, batch_idx):
        """
        Implements Heteroscedastic Forward Diffusion and Multi-domain Loss.
        """
        inputs, targets = batch
        X_0, X_0_hat = targets * 2 - 1, inputs * 2 - 1
        residual_term = X_0_hat - X_0

        # 1. Dynamic Weighting based on Subband Energy (Physics-guided)
        _, hf_hat_tuple = self.dwt(X_0_hat)
        def get_dynamic_weights(hf_tuple):
            # w = 1.0 + tanh(Energy) to prioritize heavy rain streaks
            return [1.0 + torch.tanh(torch.sqrt(torch.mean(sb**2, dim=[1,2,3], keepdim=True))) for sb in hf_tuple]
        
        dyn_weights = get_dynamic_weights(hf_hat_tuple)

        # 2. Heteroscedastic Noise Injection
        t = torch.randint(0, self.T_acc, (X_0.shape[0],), device=self.device)
        alpha_hat = self.var_scheduler.get_alphas_hat()[t].reshape(-1, 1, 1, 1)

        # Add noise to Wavelet subbands independently
        ll_res, hf_res_tuple = self.dwt(residual_term)
        r_ll_t = torch.sqrt(alpha_hat) * ll_res + torch.sqrt(1 - alpha_hat) * torch.randn_like(ll_res)
        
        r_hf_t_list = []
        for i in range(3):
            # Formula: R_t = sqrt(alpha_hat)*R_0 + w * sqrt(1-alpha_hat)*noise
            sb_t = torch.sqrt(alpha_hat) * hf_res_tuple[i] + \
                   dyn_weights[i] * torch.sqrt(1 - alpha_hat) * torch.randn_like(hf_res_tuple[i])
            r_hf_t_list.append(sb_t)

        R_t = self.iwt((r_ll_t, r_hf_t_list))
        x_t = X_0 + R_t  # Noised image for time t

        # 3. Multi-domain Loss Function
        pred_res = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)

        # (A) Wavelet Subband Loss
        p_ll, p_hf_tuple = self.dwt(pred_res)
        loss_wave = F.l1_loss(p_ll, ll_res) + sum([torch.mean(dyn_weights[i] * torch.abs(p_hf_tuple[i] - hf_res_tuple[i])) for i in range(3)])

        # (B) Spatial & Frequency Constraints
        loss_grad = self.compute_gradient_loss(pred_res, residual_term)
        loss_freq = self.compute_frequency_loss(pred_res, residual_term)

        loss = 0.6 * loss_wave + 0.2 * loss_grad + 0.2 * loss_freq
        return loss

    def generate(self, X_0_hat):
        """
        Reverse sampling using Heteroscedastic noise and Acceleration Point.
        """
        _, hf_hat_tuple = self.dwt(X_0_hat)
        dyn_weights = [1.0 + torch.tanh(torch.sqrt(torch.mean(sb**2, dim=[1,2,3], keepdim=True))) for sb in hf_hat_tuple]

        # Start from T_acc
        alpha_hat_T = self.var_scheduler.get_alphas_hat()[self.T_acc - 1]
        X_noise = X_0_hat + torch.sqrt(1 - alpha_hat_T) * torch.randn_like(X_0_hat)

        for t in range(self.T_acc - 1, -1, -1):
            t_tensor = torch.LongTensor([t]).to(self.device).expand(X_noise.shape[0])
            alpha_hat_t_minus_1 = self.var_scheduler.get_alphas_hat_t_minus_1()[t]

            # Reconstruct Heteroscedastic noise for the reverse step
            z_ll = torch.randn(X_noise.shape[0], 3, X_noise.shape[2]//2, X_noise.shape[3]//2, device=self.device)
            z_hf = [torch.randn_like(z_ll) * w for w in dyn_weights]
            z_hetero = self.iwt((z_ll, z_hf))

            # Predict residual R0
            pred_R0 = self.denoising_module(x=X_noise, time=t_tensor, input_cond=X_0_hat)
            
            # Accelerated jump: x_{t-1} = (X_in - pred_R0) + R_{t-1}
            std = torch.sqrt(1 - alpha_hat_t_minus_1)
            R_t_minus_1 = torch.sqrt(alpha_hat_t_minus_1) * pred_R0 + std * (t / self.T_acc) * z_hetero
            X_noise = (X_0_hat - pred_R0) + R_t_minus_1
            
        return X_noise

    def compute_gradient_loss(self, pred, target):
        # Sobel-based spatial consistency
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float().view(1,1,3,3).repeat(3,1,1,1)
        grad_p = torch.abs(F.conv2d(pred, kx, padding=1, groups=3)) + torch.abs(F.conv2d(pred, kx.transpose(2,3), padding=1, groups=3))
        grad_t = torch.abs(F.conv2d(target, kx, padding=1, groups=3)) + torch.abs(F.conv2d(target, kx.transpose(2,3), padding=1, groups=3))
        return F.l1_loss(grad_p, grad_t)

    def compute_frequency_loss(self, pred, target):
        # FFT-based spectral consistency
        return F.l1_loss(torch.abs(torch.fft.rfft2(pred.float())), torch.abs(torch.fft.rfft2(target.float())))