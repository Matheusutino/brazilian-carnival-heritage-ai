"""Image upscaling module using Real-ESRGAN, ESRGAN, SwinIR, EDSR and classical interpolation methods."""
from pathlib import Path
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm


class ImageUpscaler:
    """Image upscaler supporting deep learning models and classical interpolation methods."""

    # Classical interpolation methods (no model loading required)
    CLASSICAL_METHODS = ['nearest', 'bilinear', 'bicubic', 'lanczos']

    def __init__(
        self,
        model: str = 'realesrgan',
        scale: int = 2,
        device: str = 'cuda',
        tile_size: Optional[int] = None,
        tile_pad: int = 10
    ):
        """Initialize image upscaler.

        Args:
            model: Upscaling method ('realesrgan', 'esrgan', 'swinir', 'edsr', 'nearest',
                   'bilinear', 'bicubic', 'lanczos').
            scale: Upscaling factor (2 or 4 for deep learning models, any integer for classical methods).
            device: Device to run on ('cpu' or 'cuda') - only for deep learning models.
        """
        self.model_name = model
        self.scale = scale
        self.device = device
        self.upscaler = None
        self.tile_pad = tile_pad
        self._user_tile_override = tile_size is not None

        if tile_size is None:
            default_tile = 512 if self.model_name not in self.CLASSICAL_METHODS and self.device == 'cuda' else 0
        else:
            default_tile = max(tile_size, 0)

        self.tile_size = default_tile
        self._current_tile_size = default_tile
        self._min_tile_size = 64

        # Only load model for deep learning methods
        if self.model_name not in self.CLASSICAL_METHODS:
            self._load_model()

    def _load_model(self):
        """Load the upscaling model based on backend configuration."""
        print(f"Loading model: {self.model_name} (scale: {self.scale}x)...")

        if self.model_name == 'realesrgan':
            self._load_realesrgan()
        elif self.model_name == 'esrgan':
            self._load_esrgan()
        elif self.model_name == 'swinir':
            self._load_swinir()
        elif self.model_name == 'edsr':
            self._load_edsr()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        print("✓ Model loaded")

    def _load_realesrgan(self):
        """Load Real-ESRGAN model.

        Real-ESRGAN is a practical image restoration algorithm for general images.
        It extends ESRGAN by training with a better degradation process and dataset.
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "Real-ESRGAN dependencies not found. Install with: "
                "pip install realesrgan basicsr"
            )

        # Select model path based on scale
        if self.scale == 2:
            model_name = 'RealESRGAN_x2plus'
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            num_block = 23  # x2plus uses 23 blocks
        elif self.scale == 4:
            model_name = 'RealESRGAN_x4plus'
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            num_block = 23  # x4plus uses 23 blocks
        else:
            raise ValueError(f"Real-ESRGAN only supports scale 2 or 4, got {self.scale}")

        # RRDBNet architecture (23 blocks, 64 channels)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=self.scale)

        # Initialize Real-ESRGAN upsampler
        self.upscaler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=True if 'cuda' in self.device else False,  # FP16 for faster inference on GPU
            device=self.device
        )
        self._set_tile_size(self.tile_size)

    def _load_esrgan(self):
        """Load ESRGAN model.

        ESRGAN (Enhanced Super-Resolution GAN) improves upon SRGAN with
        better network architecture and perceptual loss.
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "ESRGAN dependencies not found. Install with: "
                "pip install realesrgan basicsr"
            )

        # ESRGAN model path
        if self.scale == 4:
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
        else:
            raise ValueError(f"ESRGAN only supports scale 4, got {self.scale}")

        # RRDBNet architecture for ESRGAN (23 blocks, 64 channels)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)

        # Use RealESRGANer wrapper for consistent interface
        self.upscaler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=True if 'cuda' in self.device else False,
            device=self.device
        )
        self._set_tile_size(self.tile_size)

    def _load_swinir(self):
        """Load SwinIR model.

        SwinIR uses Swin Transformer for image restoration.
        It achieves better results than CNN-based methods on various tasks.
        """
        try:
            from basicsr.archs.swinir_arch import SwinIR
        except ImportError:
            raise ImportError(
                "SwinIR dependencies not found. Install with: "
                "pip install basicsr"
            )

        # Select model configuration based on scale
        if self.scale == 2:
            model_path = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
            window_size = 8
            num_feat = 180
            depths = [6, 6, 6, 6, 6, 6]
            num_heads = [6, 6, 6, 6, 6, 6]
        elif self.scale == 4:
            model_path = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
            window_size = 8
            num_feat = 180
            depths = [6, 6, 6, 6, 6, 6]
            num_heads = [6, 6, 6, 6, 6, 6]
        else:
            raise ValueError(f"SwinIR only supports scale 2 or 4, got {self.scale}")

        # Initialize SwinIR model
        model = SwinIR(
            upscale=self.scale,
            in_chans=3,
            img_size=64,
            window_size=window_size,
            img_range=1.,
            depths=depths,
            embed_dim=num_feat,
            num_heads=num_heads,
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

        # Load pretrained weights
        import torch.hub as hub
        checkpoint = hub.load_state_dict_from_url(model_path, map_location=self.device, progress=True)

        # Handle different checkpoint formats
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        elif 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        model.eval()
        model = model.to(self.device)

        self.upscaler = model

    def _load_edsr(self):
        """Load EDSR model from HuggingFace.

        EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution)
        is a deep learning model that achieves state-of-the-art performance.
        Using eugenesiow/edsr-base from HuggingFace.
        """
        try:
            from super_image import EdsrModel
        except ImportError:
            raise ImportError(
                "EDSR dependencies not found. Install with: "
                "pip install super-image"
            )

        if self.scale not in [2, 3, 4]:
            raise ValueError(f"EDSR only supports scale 2, 3, or 4, got {self.scale}")

        # Load pretrained EDSR model from HuggingFace
        model_name = f"eugenesiow/edsr-base"

        print(f"✓ Loading EDSR from HuggingFace: {model_name}")
        model = EdsrModel.from_pretrained(model_name, scale=self.scale)

        # Remove DataParallel wrapper if present
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.eval()
        model = model.to(self.device)

        self.upscaler = model

    def _set_tile_size(self, tile_size: int):
        """Update the internal tiling configuration for deep learning models."""
        self._current_tile_size = tile_size

        if self.upscaler is None:
            return

        # Classical methods and SwinIR don't use dynamic tiling
        if self.model_name in self.CLASSICAL_METHODS or self.model_name == 'swinir':
            return

        # EDSR uses internal tiling via _process_edsr_tiles
        if self.model_name == 'edsr':
            return

        # Real-ESRGAN and ESRGAN have built-in tiling
        if hasattr(self.upscaler, 'tile'):
            self.upscaler.tile = tile_size

        if hasattr(self.upscaler, 'tile_pad'):
            self.upscaler.tile_pad = self.tile_pad

    def _prepare_next_tile_size(self) -> Optional[int]:
        """Compute the next tile size to try when a CUDA OOM happens."""
        if self.model_name in self.CLASSICAL_METHODS or self.model_name == 'swinir':
            return None

        current_tile = self._current_tile_size or 0

        if current_tile == 0:
            if self._user_tile_override and self.tile_size == 0:
                return None
            return max(self._min_tile_size, 512)

        if current_tile <= self._min_tile_size:
            return None

        next_tile = max(self._min_tile_size, current_tile // 2)

        if next_tile >= current_tile:
            next_tile = current_tile - self._min_tile_size

        if next_tile <= 0 or next_tile >= current_tile:
            return None

        return next_tile

    def _handle_cuda_oom_with_tiling(self, image_name: str) -> bool:
        """Adjust tiling to retry on GPU after a CUDA OOM."""
        if self.device != 'cuda':
            return False

        next_tile = self._prepare_next_tile_size()
        if next_tile is None:
            return False

        print(f"\n⚠ CUDA out of memory for {image_name}. Retrying on GPU with tile size {next_tile}...")
        self._set_tile_size(next_tile)
        torch.cuda.empty_cache()
        return True

    def _upscale_with_swinir(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using SwinIR model.

        Args:
            image: Input image as numpy array (BGR, uint8).

        Returns:
            Upscaled image as numpy array (BGR, uint8).
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.upscaler(img_tensor)

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        # Convert RGB back to BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output_bgr

    def _upscale_with_edsr(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using EDSR model.

        Args:
            image: Input image as numpy array (BGR, uint8).

        Returns:
            Upscaled image as numpy array (BGR, uint8).
        """
        from PIL import Image
        import torchvision.transforms as transforms

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get actual device from model
        model_device = next(self.upscaler.parameters()).device

        # Check if we need tiling (for large images on GPU)
        h, w = img_rgb.shape[:2]
        use_tiling = (
            model_device.type == 'cuda' and
            self._current_tile_size > 0 and
            (h > self._current_tile_size or w > self._current_tile_size)
        )

        if use_tiling:
            # Process in tiles
            return self._process_edsr_tiles(img_rgb, model_device)
        else:
            # Process entire image at once
            # Convert to PIL Image
            pil_image = Image.fromarray(img_rgb)

            # Convert PIL Image to tensor
            transform = transforms.ToTensor()
            img_tensor = transform(pil_image).unsqueeze(0).to(model_device)

            # Upscale using EDSR
            with torch.no_grad():
                output_tensor = self.upscaler(img_tensor)

            # Convert tensor back to numpy array
            output_tensor = output_tensor.squeeze(0).cpu()
            output_np = transforms.ToPILImage()(output_tensor)
            output_np = np.array(output_np)

            # Convert RGB back to BGR
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

            return output_bgr

    def _process_edsr_tiles(self, image: np.ndarray, device: torch.device) -> np.ndarray:
        """Process EDSR in tiles for large images.

        Args:
            image: RGB image as numpy array.
            device: Device to use for processing.

        Returns:
            Processed image as numpy array (RGB, uint8).
        """
        import torchvision.transforms as transforms

        h, w = image.shape[:2]
        tile_size = self._current_tile_size
        pad = 10  # Padding to avoid edge artifacts

        # Create output array
        output_h, output_w = h * self.scale, w * self.scale
        output = np.zeros((output_h, output_w, 3), dtype=np.float32)

        transform = transforms.ToTensor()

        # Process tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Extract tile with padding
                y_start = max(0, y - pad)
                y_end = min(h, y + tile_size + pad)
                x_start = max(0, x - pad)
                x_end = min(w, x + tile_size + pad)

                tile = image[y_start:y_end, x_start:x_end]

                # Convert tile to tensor
                tile_pil = Image.fromarray(tile)
                tile_tensor = transform(tile_pil).unsqueeze(0).to(device)

                # Process tile
                with torch.no_grad():
                    tile_output = self.upscaler(tile_tensor)

                tile_output = tile_output.squeeze(0).cpu()
                tile_output_pil = transforms.ToPILImage()(tile_output)
                tile_output_np = np.array(tile_output_pil).astype(np.float32)

                # Calculate actual tile region (removing padding)
                pad_y_start = (y - y_start) * self.scale
                pad_x_start = (x - x_start) * self.scale
                actual_h = min(tile_size * self.scale, output_h - y * self.scale)
                actual_w = min(tile_size * self.scale, output_w - x * self.scale)

                # Place tile in output (without padding)
                output[y * self.scale:y * self.scale + actual_h,
                       x * self.scale:x * self.scale + actual_w] = tile_output_np[
                    pad_y_start:pad_y_start + actual_h,
                    pad_x_start:pad_x_start + actual_w
                ]

        # Convert to uint8
        output = output.clip(0, 255).astype(np.uint8)

        # Convert RGB back to BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output_bgr

    def _upscale_with_classical(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using classical interpolation methods.

        Args:
            image: Input image as numpy array (BGR, uint8).

        Returns:
            Upscaled image as numpy array (BGR, uint8).
        """
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale, w * self.scale

        # Map method names to OpenCV interpolation flags
        interpolation_map = {
            'nearest': cv2.INTER_NEAREST,      # Nearest neighbor
            'bilinear': cv2.INTER_LINEAR,      # Bilinear interpolation
            'bicubic': cv2.INTER_CUBIC,        # Bicubic interpolation
            'lanczos': cv2.INTER_LANCZOS4      # Lanczos interpolation (high quality)
        }

        if self.model_name not in interpolation_map:
            raise ValueError(f"Unknown classical method: {self.model_name}")

        interpolation = interpolation_map[self.model_name]

        # Perform upscaling using OpenCV resize
        output = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        return output

    def _run_model_inference(self, image: np.ndarray) -> np.ndarray:
        """Run the appropriate inference backend for the configured model."""
        if self.model_name in self.CLASSICAL_METHODS:
            return self._upscale_with_classical(image)

        if self.model_name == 'swinir':
            return self._upscale_with_swinir(image)

        if self.model_name == 'edsr':
            return self._upscale_with_edsr(image)

        # Real-ESRGAN and ESRGAN
        output, _ = self.upscaler.enhance(image, outscale=self.scale)
        return output

    def _restore_gpu_after_cpu(self):
        """Move the upscaler back to GPU after a CPU fallback."""
        if self.device != 'cuda':
            return

        if self.model_name in ['swinir', 'edsr']:
            self.upscaler = self.upscaler.cuda()
        elif self.model_name not in self.CLASSICAL_METHODS:
            self.upscaler.device = torch.device('cuda')
            if hasattr(self.upscaler, 'model'):
                self.upscaler.model = self.upscaler.model.cuda()

        torch.cuda.empty_cache()

    def _process_on_cpu(self, image: np.ndarray, image_name: str) -> np.ndarray:
        """Fallback processing on CPU when GPU retries fail."""
        print(f"\n⚠ CUDA out of memory for {image_name}. Processing this image on CPU...")

        try:
            if self.model_name in ['swinir', 'edsr']:
                self.upscaler = self.upscaler.cpu()
            elif self.model_name not in self.CLASSICAL_METHODS:
                self.upscaler.device = torch.device('cpu')
                if hasattr(self.upscaler, 'model'):
                    self.upscaler.model = self.upscaler.model.cpu()

            torch.cuda.empty_cache()

            output = self._run_model_inference(image)

        except Exception as cpu_error:
            self._restore_gpu_after_cpu()
            raise RuntimeError(f"CPU fallback failed: {cpu_error}") from cpu_error

        self._restore_gpu_after_cpu()
        print(f"✓ Processed on CPU: {image_name}. Model moved back to GPU...")

        return output

    def upscale_image(self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Dict:
        """Upscale a single image.

        Args:
            image_path: Path to the input image file.
            output_path: Path to save the upscaled image. If None, auto-generates path.

        Returns:
            Dictionary containing upscaling results and metadata.
        """
        image_path = Path(image_path)

        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'model': self.model_name,
            'scale': self.scale,
            'status': 'pending'
        }

        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")

            h, w = img.shape[:2]
            result['original_size'] = (w, h)

            # Skip upscaling if image is already Full HD or larger
            if w >= 1920 and h >= 1080:
                # Just copy the original image to output
                if output_path is None:
                    output_dir = image_path.parent / f'upscaled_{self.model_name}_{self.scale}x'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{image_path.stem}_{self.model_name}_{self.scale}x{image_path.suffix}"
                else:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(output_path), img)

                result['upscaled_size'] = (w, h)
                result['output_path'] = str(output_path)
                result['status'] = 'success'
                result['skipped_upscale'] = True

                return result

            # Check if upscaled image will be too large for GPU (>2GB estimated)
            # This prevents OOM during tile merging phase
            upscaled_pixels = (h * self.scale) * (w * self.scale)
            estimated_mb = (upscaled_pixels * 3 * 4) / (1024**2)  # RGB, float32
            force_cpu = (
                self.device == 'cuda'
                and self.model_name not in self.CLASSICAL_METHODS
                and estimated_mb > 2000  # Conservative threshold
            )

            output = None
            fallback_cpu_used = False

            if force_cpu:
                print(f"\n💡 Image {image_path.name} ({w}x{h}) will be {w*self.scale}x{h*self.scale} after upscale (~{estimated_mb:.0f}MB)")
                print(f"   Processing directly on CPU to avoid GPU memory issues during tile merge...")
                output = self._process_on_cpu(img, image_path.name)
                fallback_cpu_used = True
            else:
                while True:
                    try:
                        output = self._run_model_inference(img)
                        break

                    except RuntimeError as runtime_error:
                        error_message = str(runtime_error)
                        is_oom = 'cuda out of memory' in error_message.lower() or 'out of memory' in error_message.lower()

                        if is_oom and self.device == 'cuda' and self.model_name not in self.CLASSICAL_METHODS:
                            if self._handle_cuda_oom_with_tiling(image_path.name):
                                continue

                            try:
                                output = self._process_on_cpu(img, image_path.name)
                                fallback_cpu_used = True
                                break
                            except RuntimeError as cpu_error:
                                result['status'] = 'error'
                                result['error'] = str(cpu_error)
                                return result

                        result['status'] = 'error'
                        result['error'] = error_message
                        return result

                    except Exception as generic_error:
                        result['status'] = 'error'
                        result['error'] = str(generic_error)
                        return result

            h_out, w_out = output.shape[:2]
            result['upscaled_size'] = (w_out, h_out)

            if output_path is None:
                output_dir = image_path.parent / f'upscaled_{self.model_name}_{self.scale}x'
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}_{self.model_name}_{self.scale}x{image_path.suffix}"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(output_path), output)

            result['output_path'] = str(output_path)
            result['status'] = 'success'

            if fallback_cpu_used:
                result['fallback_cpu'] = True

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def upscale_batch(
        self,
        image_paths: List[Path],
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """Upscale multiple images.

        Args:
            image_paths: List of paths to input image files.
            output_dir: Directory to save upscaled images. If None, creates subdirectory next to each image.
            verbose: If True, prints progress information.

        Returns:
            List of dictionaries containing upscaling results.
        """
        results = []

        if verbose:
            print(f"\nUpscaling {len(image_paths)} images...")
            if self.model_name in self.CLASSICAL_METHODS:
                print(f"Method: {self.model_name} (classical interpolation), Scale: {self.scale}x")
            else:
                print(f"Model: {self.model_name}, Scale: {self.scale}x, Device: {self.device}")

        iterator = tqdm(image_paths, desc="Upscaling images") if verbose else image_paths

        for img_path in iterator:
            # Generate output path
            if output_dir is not None:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = output_dir_path / f"{img_path.stem}_{self.model_name}_{self.scale}x{img_path.suffix}"
            else:
                output_path = None

            # Upscale image
            result = self.upscale_image(img_path, output_path)
            results.append(result)

            if verbose and result['status'] != 'success':
                print(f"✗ {img_path.name}: {result.get('error', 'Unknown error')}")

        if verbose:
            success = sum(1 for r in results if r['status'] == 'success')
            cpu_fallback = sum(1 for r in results if r.get('fallback_cpu', False))
            print(f"\n✓ Complete: {success}/{len(results)} successful")
            if cpu_fallback > 0:
                print(f"💡 {cpu_fallback} image(s) processed on CPU due to GPU memory limitations")

        return results
