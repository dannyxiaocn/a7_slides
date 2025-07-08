"""
Image Quality Assessment Module
Implements various Full-Reference (FR) and No-Reference (NR) IQA measures
for evaluating deblurring and inpainting results from Module 2.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage import filters, feature, measure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class ImageQualityAssessment:
    """
    Comprehensive Image Quality Assessment class implementing various FR and NR metrics.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Full-Reference (FR) Metrics
    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        return float(psnr(original, reconstructed, data_range=1.0))
    
    def calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        if len(original.shape) == 3:
            return float(ssim(original, reconstructed, channel_axis=2, data_range=1.0))
        else:
            return float(ssim(original, reconstructed, data_range=1.0))
    
    def calculate_fsim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Feature Similarity Index (FSIM)
        Based on phase congruency and gradient magnitude
        """
        def phase_congruency(img):
            """Simplified phase congruency calculation"""
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            
            # Calculate gradients
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            # Phase congruency approximation using structure tensor
            Ixx = gx * gx
            Ixy = gx * gy
            Iyy = gy * gy
            
            # Gaussian smoothing
            sigma = 1.0
            Ixx = ndimage.gaussian_filter(Ixx, sigma)
            Ixy = ndimage.gaussian_filter(Ixy, sigma)
            Iyy = ndimage.gaussian_filter(Iyy, sigma)
            
            # Eigenvalues of structure tensor
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            
            lambda1 = trace/2 + np.sqrt((trace/2)**2 - det + 1e-10)
            lambda2 = trace/2 - np.sqrt((trace/2)**2 - det + 1e-10)
            
            # Phase congruency approximation
            pc = lambda2 / (lambda1 + 1e-10)
            
            return pc, gradient_mag
        
        # Calculate phase congruency and gradient magnitude for both images
        pc1, gm1 = phase_congruency(original)
        pc2, gm2 = phase_congruency(reconstructed)
        
        # FSIM calculation
        # Similarity measures
        T1, T2, T3, T4 = 0.85, 0.85, 0.25, 0.25  # Thresholds
        
        # Phase congruency similarity
        pc_sim = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)
        
        # Gradient magnitude similarity
        gm_sim = (2 * gm1 * gm2 + T2) / (gm1**2 + gm2**2 + T2)
        
        # Combined similarity
        sl = pc_sim * gm_sim
        
        # Weighting based on maximum phase congruency
        pcm = np.maximum(pc1, pc2)
        
        # FSIM index
        fsim = np.sum(sl * pcm) / (np.sum(pcm) + 1e-10)
        
        return float(fsim)
    
    def calculate_vif(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Visual Information Fidelity (VIF)
        Simplified implementation based on mutual information
        """
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            recon_gray = cv2.cvtColor((reconstructed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = (original * 255).astype(np.uint8)
            recon_gray = (reconstructed * 255).astype(np.uint8)
        
        # Calculate normalized mutual information
        def mutual_information(img1, img2):
            # Joint histogram
            hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256, range=[[0, 256], [0, 256]])
            
            # Normalize
            hist_2d = hist_2d / np.sum(hist_2d)
            
            # Marginal histograms
            hist1 = np.sum(hist_2d, axis=1)
            hist2 = np.sum(hist_2d, axis=0)
            
            # Calculate mutual information
            mi = 0
            for i in range(256):
                for j in range(256):
                    if hist_2d[i, j] > 0:
                        mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist1[i] * hist2[j] + 1e-10))
            
            return mi
        
        # Calculate entropy of original image
        hist_orig = np.histogram(orig_gray, bins=256, range=(0, 256))[0]
        hist_orig = hist_orig / np.sum(hist_orig)
        entropy_orig = entropy(hist_orig + 1e-10, base=2)
        
        # Calculate mutual information
        mi = mutual_information(orig_gray, recon_gray)
        
        # VIF approximation
        vif = mi / (entropy_orig + 1e-10)
        
        return float(vif)
    
    # No-Reference (NR) Metrics
    def calculate_brisque(self, image: np.ndarray) -> float:
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
        Simplified implementation based on natural scene statistics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            gray = (image * 255).astype(np.float64)
        
        # Mean subtracted contrast normalized (MSCN) coefficients
        mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(gray * gray, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        
        # MSCN coefficients
        mscn = (gray - mu) / (sigma + 1e-10)
        
        # Calculate moments
        alpha = np.mean(mscn**2)
        beta = np.mean(mscn**4) / (alpha**2)
        
        # Pairwise products
        # Horizontal
        h_mscn = mscn[:, :-1] * mscn[:, 1:]
        # Vertical
        v_mscn = mscn[:-1, :] * mscn[1:, :]
        # Main diagonal
        d1_mscn = mscn[:-1, :-1] * mscn[1:, 1:]
        # Anti-diagonal
        d2_mscn = mscn[:-1, 1:] * mscn[1:, :-1]
        
        # Extract features (simplified)
        features = [
            alpha, beta,
            np.mean(h_mscn), np.var(h_mscn),
            np.mean(v_mscn), np.var(v_mscn),
            np.mean(d1_mscn), np.var(d1_mscn),
            np.mean(d2_mscn), np.var(d2_mscn)
        ]
        
        # BRISQUE score approximation (lower is better)
        brisque_score = np.mean(np.abs(features))
        
        return float(brisque_score)
    
    def calculate_niqe(self, image: np.ndarray) -> float:
        """
        Calculate NIQE (Natural Image Quality Evaluator)
        Simplified implementation
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            gray = (image * 255).astype(np.float64)
        
        # Local mean and variance
        mu = cv2.GaussianBlur(gray, (7, 7), 1.5)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(gray * gray, (7, 7), 1.5)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        
        # Structural degradation
        structdis = (gray - mu) / (sigma + 1e-10)
        
        # Calculate features
        features = [
            np.mean(structdis),
            np.var(structdis),
            np.mean(np.abs(structdis)),
            np.mean(structdis**3),
            np.mean(structdis**4)
        ]
        
        # NIQE score approximation
        niqe_score = np.std(features)
        
        return float(niqe_score)
    
    def calculate_piqe(self, image: np.ndarray) -> float:
        """
        Calculate PIQE (Perception-based Image Quality Evaluator)
        Based on block-wise analysis
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Block size
        block_size = 16
        h, w = gray.shape
        
        # Calculate block-wise variance
        block_variances = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))
        
        block_variances = np.array(block_variances)
        
        # PIQE calculation
        # Activity threshold
        T = 0.1 * np.max(block_variances)
        
        # Distortion computation
        active_blocks = block_variances[block_variances > T]
        
        if len(active_blocks) > 0:
            piqe_score = np.mean(active_blocks)
        else:
            piqe_score = np.mean(block_variances)
        
        return float(piqe_score)
    
    def comprehensive_evaluation(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive quality assessment using all implemented metrics
        """
        results = {}
        
        # Full-Reference metrics
        results['PSNR'] = self.calculate_psnr(original, reconstructed)
        results['SSIM'] = self.calculate_ssim(original, reconstructed)
        results['FSIM'] = self.calculate_fsim(original, reconstructed)
        results['VIF'] = self.calculate_vif(original, reconstructed)
        
        # No-Reference metrics (on reconstructed image)
        results['BRISQUE'] = self.calculate_brisque(reconstructed)
        results['NIQE'] = self.calculate_niqe(reconstructed)
        results['PIQE'] = self.calculate_piqe(reconstructed)
        
        return results
    
    def create_degraded_versions(self, image: np.ndarray, target_psnr: float = 25.0, 
                               target_ssim: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Create different degraded versions of an image with similar PSNR and SSIM values
        """
        degraded_images = {}
        
        # Original image
        original = image.copy()
        
        # Method 1: Gaussian Blur
        for sigma in np.linspace(0.5, 3.0, 20):
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            psnr_val = self.calculate_psnr(original, blurred)
            ssim_val = self.calculate_ssim(original, blurred)
            
            if abs(psnr_val - target_psnr) < 1.0:
                degraded_images[f'blur_psnr_{psnr_val:.1f}'] = blurred
                break
        
        # Method 2: Gaussian Noise
        for noise_std in np.linspace(0.01, 0.1, 20):
            noise = np.random.normal(0, noise_std, image.shape)
            noisy = np.clip(image + noise, 0, 1)
            psnr_val = self.calculate_psnr(original, noisy)
            ssim_val = self.calculate_ssim(original, noisy)
            
            if abs(psnr_val - target_psnr) < 1.0:
                degraded_images[f'noise_psnr_{psnr_val:.1f}'] = noisy
                break
        
        # Method 3: JPEG-like compression artifacts
        for quality in range(10, 100, 5):
            # Simulate compression by reducing bit depth and adding quantization noise
            compressed = (image * 255).astype(np.uint8)
            # Add quantization noise
            quantization_step = (100 - quality) / 10.0
            quantized = np.round(compressed / quantization_step) * quantization_step
            quantized = np.clip(quantized, 0, 255) / 255.0
            
            psnr_val = self.calculate_psnr(original, quantized)
            ssim_val = self.calculate_ssim(original, quantized)
            
            if abs(psnr_val - target_psnr) < 1.0:
                degraded_images[f'compression_psnr_{psnr_val:.1f}'] = quantized
                break
        
        # Similar process for SSIM
        # Method 4: Motion Blur for SSIM target
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel = kernel / kernel_size
        
        motion_blurred = cv2.filter2D(image, -1, kernel)
        ssim_val = self.calculate_ssim(original, motion_blurred)
        
        if abs(ssim_val - target_ssim) < 0.05:
            degraded_images[f'motion_blur_ssim_{ssim_val:.2f}'] = motion_blurred
        
        return degraded_images
    
    def analyze_background_effect(self, image_with_bg: np.ndarray, image_without_bg: np.ndarray,
                                degraded_with_bg: np.ndarray, degraded_without_bg: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Analyze the effect of background removal on quality metrics
        """
        results = {}
        
        # With background
        results['with_background'] = self.comprehensive_evaluation(image_with_bg, degraded_with_bg)
        
        # Without background
        results['without_background'] = self.comprehensive_evaluation(image_without_bg, degraded_without_bg)
        
        # Calculate differences
        results['difference'] = {}
        for metric in results['with_background']:
            diff = results['without_background'][metric] - results['with_background'][metric]
            results['difference'][metric] = diff
        
        return results 