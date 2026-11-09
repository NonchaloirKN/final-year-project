"""
AcousticSimulator: Physics-Based Ultrasound Degradation Modeling.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Simulates intraoperative 2D ultrasound slices from preoperative 2D MRI slices on the GPU in five stages:
1. Interface reflection: echo = (1 - w) * I + w * |dI/dr| / max, the intensity gradient along the beam
   direction r standing in for the acoustic-impedance mismatch at tissue boundaries (cf. the
   intensity-plus-gradient ultrasound model of Wein et al. 2008).
2. Multiplicative Rayleigh speckle (fully developed speckle envelope).
3. Beer-Lambert attenuation exp(-alpha_Np * f0 * d) along the radial depth d of the beam.
4. Curvilinear sector (fan) mask with smooth edges, apex above the top centre of the frame.
5. Per-slice min-max normalisation to [0, 1].
Setting reflection_weight=0, fan_angle_deg=None and per_slice_norm=False reproduces the Sprint 1-8a
simulator exactly (speckle -> row-wise attenuation -> batch-wide min-max).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


# Default physical constants from FCN05 System Architecture & Literature
DEFAULT_F0: float = 5.0  # Operating central frequency (MHz)
DEFAULT_C: float = 1540.0  # Speed of sound in human soft tissue (m/s)
DEFAULT_ALPHA_WM: float = 0.60  # Brain attenuation (dB/cm/MHz), Culjat et al. 2010, Table 1 (ICRU 1998)
DEFAULT_SIGMA_RAYLEIGH: float = 1.0  # Unit-scale Rayleigh envelope (Wagner et al. 1983); per-slice min-max removes the scale
LEGACY_ALPHA_WM: float = 0.452  # Sprint 1-8c value (no recorded source)
LEGACY_SIGMA_RAYLEIGH: float = 59.09  # Sprint 1-8c value (scale only)


class AcousticSimulator:
    """
    Physics-based ultrasound simulation engine with GPU tensor acceleration.

    Models ultrasound propagation through brain tissue by applying:
    1. Tissue acoustic attenuation following the Beer-Lambert decay law:
       I(l) = I_0 * exp(-alpha_Np * l * f_0)
    2. Constructive and destructive backscatter wave interference modeled
       via a Rayleigh distribution with parameter sigma.

    Parameters
    ----------
    f0 : float, optional
        Transducer central frequency in MHz, by default 5.0.
    c : float, optional
        Propagation velocity in m/s, by default 1540.0.
    alpha_wm : float, optional
        Acoustic attenuation coefficient in dB/cm/MHz, by default 0.60 (brain, Culjat et al. 2010).
    sigma_rayleigh : float, optional
        Rayleigh speckle distribution scale parameter, by default 1.0 (the output is renormalised).
    device : Optional[Union[str, torch.device]], optional
        Target compute device ('cuda' or 'cpu'). Defaults to CUDA if available.
    """

    def __init__(
        self,
        f0: float = DEFAULT_F0,
        c: float = DEFAULT_C,
        alpha_wm: float = DEFAULT_ALPHA_WM,
        sigma_rayleigh: float = DEFAULT_SIGMA_RAYLEIGH,
        device: Optional[Union[str, torch.device]] = None,
        reflection_weight: float = 0.3,
        fan_angle_deg: Optional[float] = 80.0,
        apex_offset_frac: float = 0.25,
        per_slice_norm: bool = True,
    ) -> None:
        self.f0 = float(f0)
        self.c = float(c)
        self.alpha_wm = float(alpha_wm)
        self.sigma_rayleigh = float(sigma_rayleigh)
        self.reflection_weight = float(reflection_weight)
        self.fan_angle_deg = fan_angle_deg
        self.apex_offset_frac = float(apex_offset_frac)
        self.per_slice_norm = bool(per_slice_norm)
        self._fan_cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}

        # Convert acoustic attenuation from dB/cm/MHz to Neper/cm/MHz:
        # alpha_Np = alpha_dB * ln(10) / 20
        self.alpha_np = self.alpha_wm * math.log(10.0) / 20.0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def apply_attenuation(
        self,
        image: Union[float, int, np.ndarray, torch.Tensor],
        depth_cm: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        max_depth_cm: float = 8.0,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Apply Beer-Lambert acoustic depth attenuation:
            I(l) = I_0 * exp(-alpha_Np * l * f_0)

        Parameters
        ----------
        image : Union[float, int, np.ndarray, torch.Tensor]
            Input intensity, image array, or image tensor.
        depth_cm : Optional[Union[float, np.ndarray, torch.Tensor]], optional
            Explicit penetration depth in cm. If None, depth is computed
            linearly along the vertical axis (0 to max_depth_cm).
        max_depth_cm : float, optional
            Maximum imaging depth represented by the bottom of the image, by default 8.0 cm.

        Returns
        -------
        Union[float, np.ndarray, torch.Tensor]
            Attenuated intensity or tensor matching input type and device.
        """
        # Case A: Scalar intensity with specified scalar depth
        if isinstance(image, (int, float)) and depth_cm is not None:
            decay = math.exp(-self.alpha_np * float(depth_cm) * self.f0)
            return float(image * decay)

        # Case B: PyTorch Tensor processing (GPU accelerated)
        if isinstance(image, torch.Tensor):
            if depth_cm is not None:
                if isinstance(depth_cm, (int, float)):
                    decay_factor = math.exp(-self.alpha_np * float(depth_cm) * self.f0)
                    return image * decay_factor
                elif isinstance(depth_cm, torch.Tensor):
                    decay_tensor = torch.exp(-self.alpha_np * depth_cm.to(image.device) * self.f0)
                    return image * decay_tensor

            # Compute depth map along vertical axis (height)
            h = image.shape[-2]
            depth_vector = torch.linspace(
                0.0,
                max_depth_cm,
                steps=h,
                dtype=image.dtype,
                device=image.device,
            )
            decay_vector = torch.exp(-self.alpha_np * depth_vector * self.f0)

            # Broadcast across remaining dimensions
            if image.ndim == 2:  # [H, W]
                decay_grid = decay_vector.unsqueeze(1)
            elif image.ndim == 3:  # [C, H, W]
                decay_grid = decay_vector.unsqueeze(0).unsqueeze(2)
            elif image.ndim == 4:  # [B, C, H, W]
                decay_grid = decay_vector.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            else:
                decay_grid = decay_vector

            return image * decay_grid

        # Case C: NumPy Array processing
        if isinstance(image, np.ndarray):
            if depth_cm is not None:
                decay_factor = np.exp(-self.alpha_np * float(depth_cm) * self.f0)
                return image * decay_factor

            h = image.shape[-2]
            depth_vector = np.linspace(0.0, max_depth_cm, h, dtype=np.float32)
            decay_vector = np.exp(-self.alpha_np * depth_vector * self.f0)

            if image.ndim == 2:
                decay_grid = decay_vector[:, np.newaxis]
            elif image.ndim == 3:
                decay_grid = decay_vector[np.newaxis, :, np.newaxis]
            elif image.ndim == 4:
                decay_grid = decay_vector[np.newaxis, np.newaxis, :, np.newaxis]
            else:
                decay_grid = decay_vector

            return (image * decay_grid).astype(np.float32)

        # Fallback for scalar image without depth
        return image

    def apply_rayleigh_speckle(
        self,
        image: Union[np.ndarray, torch.Tensor],
        sigma: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate Rayleigh noise and multiply over the image to simulate
        constructive and destructive ultrasound speckle wave interference.

        Rayleigh distribution with scale parameter sigma is sampled via:
            R = sqrt(Z_1^2 + Z_2^2), where Z_1, Z_2 ~ N(0, sigma^2)

        Parameters
        ----------
        image : Union[np.ndarray, torch.Tensor]
            Input image tensor or array.
        sigma : Optional[float], optional
            Rayleigh scale parameter. Defaults to self.sigma_rayleigh (1.0 by default; 59.09 only for legacy()).
        generator : Optional[torch.Generator], optional
            PyTorch random number generator for reproducible sampling.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Speckled image matching input type and device.
        """
        s = float(sigma) if sigma is not None else self.sigma_rayleigh

        if isinstance(image, torch.Tensor):
            # Generate 2 independent Gaussian distributions on image's device
            z1 = torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator) * s
            z2 = torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator) * s
            rayleigh_noise = torch.sqrt(z1**2 + z2**2 + 1e-12)
            return image * rayleigh_noise

        elif isinstance(image, np.ndarray):
            rayleigh_noise = np.random.rayleigh(scale=s, size=image.shape).astype(np.float32)
            return (image * rayleigh_noise).astype(np.float32)

        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image).__name__}")

    @classmethod
    def legacy(cls, device: Optional[Union[str, torch.device]] = None) -> "AcousticSimulator":
        """Sprint 1-8a simulator: no reflection or fan mask, batch-wide normalisation, original constants."""
        return cls(device=device, reflection_weight=0.0, fan_angle_deg=None, per_slice_norm=False,
                   alpha_wm=LEGACY_ALPHA_WM, sigma_rayleigh=LEGACY_SIGMA_RAYLEIGH)

    def config(self) -> Dict[str, object]:
        """Settings needed to rebuild this simulator (stored in checkpoints)."""
        return {"reflection_weight": self.reflection_weight, "fan_angle_deg": self.fan_angle_deg,
                "apex_offset_frac": self.apex_offset_frac, "per_slice_norm": self.per_slice_norm,
                "f0": self.f0, "alpha_wm": self.alpha_wm, "sigma_rayleigh": self.sigma_rayleigh}

    def fan_geometry(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """
        Curvilinear sector geometry for an h x w frame (cached): apex at (x, y) = ((w - 1) / 2, -r0) above
        the top centre with r0 = apex_offset_frac * h, outer arc at radius r0 + h - 1 (bottom centre),
        half-aperture fan_angle_deg / 2. Row index increases with depth.

        Returns:
            mask: smooth [0, 1] sector mask (1-pixel logistic edges), shape [h, w].
            depth_frac: radial depth from the apex arc, normalised to [0, 1] over the sector, shape [h, w].
            ux, uy: unit vector of the beam (radial) direction at each pixel, shape [h, w].
        """
        key = (h, w, str(device), dtype, self.fan_angle_deg, self.apex_offset_frac)
        if key not in self._fan_cache:
            r0 = self.apex_offset_frac * h
            r_max = r0 + h - 1
            ys = torch.arange(h, device=device, dtype=dtype).unsqueeze(1) + r0
            xs = torch.arange(w, device=device, dtype=dtype).unsqueeze(0) - (w - 1) / 2.0
            r = torch.sqrt(xs ** 2 + ys ** 2)
            phi = torch.atan2(xs, ys)
            half = math.radians(self.fan_angle_deg) / 2.0
            inside_angle = r * torch.sin(half - phi.abs())
            mask = torch.sigmoid(inside_angle) * torch.sigmoid(r_max - r)
            depth_frac = ((r - r0) / (r_max - r0)).clamp(0.0, 1.0)
            self._fan_cache[key] = {"mask": mask, "depth_frac": depth_frac, "ux": xs / r, "uy": ys / r}
        return self._fan_cache[key]

    def simulate_us(
        self,
        mri_slice: Union[np.ndarray, torch.Tensor],
        max_depth_cm: float = 8.0,
        normalize: bool = True,
        generator: Optional[torch.Generator] = None,
        return_stages: bool = False,
    ) -> Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate an intraoperative 2D ultrasound slice from a preoperative 2D MRI slice.

        Pipeline (see module docstring): reflection -> Rayleigh speckle -> Beer-Lambert attenuation ->
        sector mask -> min-max normalisation (per slice by default).

        Parameters
        ----------
        mri_slice : Union[np.ndarray, torch.Tensor]
            Input MRI slice(s) scaled to [0, 1], shape (..., H, W).
        max_depth_cm : float, optional
            Acoustic depth represented by the full depth of the frame, by default 8.0 cm.
        normalize : bool, optional
            Whether to normalize the output back to [0, 1], by default True.
        generator : Optional[torch.Generator], optional
            RNG for the speckle field.
        return_stages : bool, optional
            If True, return a dict of intermediate stages (for figures).

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Simulated ultrasound frame matching the input type.
        """
        is_numpy = isinstance(mri_slice, np.ndarray)
        x = torch.from_numpy(np.asarray(mri_slice, dtype=np.float32)) if is_numpy else mri_slice
        h, w = x.shape[-2], x.shape[-1]
        fan = None
        if self.fan_angle_deg is not None:
            fan = self.fan_geometry(h, w, x.device, x.dtype)

        # Stage 1: interface reflection from the intensity gradient along the beam direction
        if self.reflection_weight > 0.0:
            pad = torch.nn.functional.pad(x.reshape(-1, 1, h, w), (1, 1, 1, 1), mode="replicate").reshape(*x.shape[:-2], h + 2, w + 2)
            gy = 0.5 * (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1])
            gx = 0.5 * (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2])
            grad = (gx * fan["ux"] + gy * fan["uy"]).abs() if fan is not None else gy.abs()
            grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            echo = (1.0 - self.reflection_weight) * x + self.reflection_weight * grad
        else:
            grad = torch.zeros_like(x)
            echo = x

        # Stage 2: multiplicative Rayleigh speckle
        speckled = self.apply_rayleigh_speckle(echo, generator=generator)

        # Stage 3: Beer-Lambert attenuation along the radial depth (rows, without a fan)
        if fan is not None:
            gain = torch.exp(-self.alpha_np * self.f0 * max_depth_cm * fan["depth_frac"])
            attenuated = speckled * gain
        else:
            gain = self.apply_attenuation(torch.ones(h, w, device=x.device, dtype=x.dtype), max_depth_cm=max_depth_cm)
            attenuated = self.apply_attenuation(speckled, max_depth_cm=max_depth_cm)

        # Stage 4: curvilinear sector mask
        masked = attenuated * fan["mask"] if fan is not None else attenuated

        # Stage 5: min-max normalisation, per slice (default) or over the whole input (legacy)
        out = masked
        if normalize:
            if self.per_slice_norm:
                i_min = masked.amin(dim=(-2, -1), keepdim=True)
                i_max = masked.amax(dim=(-2, -1), keepdim=True)
                out = ((masked - i_min) / (i_max - i_min + 1e-8)).clamp(0.0, 1.0)
            else:
                i_min, i_max = masked.min(), masked.max()
                out = torch.zeros_like(masked) if torch.isclose(i_min, i_max) else \
                    ((masked - i_min) / (i_max - i_min + 1e-8)).clamp(0.0, 1.0)

        if return_stages:
            return {"input": x, "reflection": grad, "echo": echo, "speckled": speckled, "gain": gain,
                    "mask": fan["mask"] if fan is not None else torch.ones_like(gain), "output": out}
        if is_numpy:
            return out.cpu().numpy().astype(np.float32)
        return out

    __call__ = simulate_us
