"""Image quality assessment using fast OpenCV-based metrics.

This module implements efficient no-reference image quality metrics using OpenCV.
These metrics are optimized for speed while providing useful quality indicators.
"""
from pathlib import Path
from typing import Dict, Optional, Union, List
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def _evaluate_single_image(image_path: Path, original_images_dir: Optional[Path]) -> Dict:
    """Helper function to evaluate a single image (for multiprocessing).

    Args:
        image_path: Path to image to evaluate.
        original_images_dir: Directory containing original images.

    Returns:
        Dictionary with evaluation results.
    """
    # Create a temporary evaluator instance
    evaluator = ImageQualityEvaluator(original_images_dir=original_images_dir)
    return evaluator.evaluate_image(image_path)


class ImageQualityEvaluator:
    """Evaluator for fast image quality metrics using OpenCV."""

    def __init__(self, original_images_dir: Optional[Union[str, Path]] = None):
        """Initialize quality evaluator.

        Args:
            original_images_dir: Directory containing original images for self-consistency.
                                Default: dataset/downloaded
        """
        if original_images_dir is None:
            # Default to dataset/downloaded
            project_root = Path(__file__).parent.parent.parent.parent
            self.original_images_dir = project_root / "dataset" / "downloaded"
        else:
            self.original_images_dir = Path(original_images_dir)

        if not self.original_images_dir.exists():
            print(f"Warning: Original images directory not found: {self.original_images_dir}")
            self.original_images_dir = None

    def _load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load image using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as numpy array (BGR format), or None if error.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Error loading image: {image_path}")
                return None
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def calculate_sharpness(self, img: np.ndarray) -> float:
        """Calculate sharpness using Variance of Laplacian.

        Higher values indicate sharper images.

        Args:
            img: Input image (BGR format).

        Returns:
            Sharpness score.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def calculate_edge_strength(self, img: np.ndarray) -> float:
        """Calculate edge strength using Sobel operators.

        Higher values indicate stronger edges (more detail).

        Args:
            img: Input image (BGR format).

        Returns:
            Mean edge strength.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return edge_magnitude.mean()

    def calculate_contrast(self, img: np.ndarray) -> float:
        """Calculate contrast using standard deviation.

        Higher values indicate higher contrast.

        Args:
            img: Input image (BGR format).

        Returns:
            Contrast score (standard deviation of grayscale image).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.std()

    def calculate_self_consistency(self, img_original: np.ndarray,
                                   img_upscaled: np.ndarray) -> float:
        """Calculate self-consistency using PSNR.

        Measures how well the upscaled image preserves the original when downsampled.
        Higher values indicate better consistency.

        Args:
            img_original: Original (low-resolution) image.
            img_upscaled: Upscaled image.

        Returns:
            PSNR score (dB).
        """
        h, w = img_original.shape[:2]
        img_down = cv2.resize(img_upscaled, (w, h), interpolation=cv2.INTER_AREA)

        # Ensure same size
        if img_original.shape != img_down.shape:
            return 0.0

        try:
            psnr = cv2.PSNR(img_original, img_down)
            return psnr
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0.0

    def calculate_artifacts(self, img: np.ndarray) -> Dict[str, float]:
        """Detect common upscaling artifacts.

        Args:
            img: Input image (BGR format).

        Returns:
            Dictionary with artifact scores:
            - oversharp: Over-sharpening indicator (99th percentile of Laplacian)
            - ringing: Ringing artifact indicator (std of Sobel)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Over-sharpening detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        oversharp = np.percentile(np.abs(laplacian), 99)

        # Ringing detection
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        ringing = np.std(sobel)

        return {
            'oversharp': float(oversharp),
            'ringing': float(ringing)
        }

    def evaluate_image(self, image_path: Union[str, Path]) -> Dict:
        """Evaluate image quality using all available metrics.

        Args:
            image_path: Path to the image file to evaluate.

        Returns:
            Dictionary containing all quality scores and metadata.
        """
        image_path = Path(image_path)

        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'status': 'pending'
        }

        try:
            # Load image
            img = self._load_image(image_path)
            if img is None:
                result['status'] = 'error'
                result['error'] = 'Failed to load image'
                return result

            # Calculate all metrics
            result['sharpness'] = self.calculate_sharpness(img)
            result['edge_strength'] = self.calculate_edge_strength(img)
            result['contrast'] = self.calculate_contrast(img)

            # Artifact detection
            artifacts = self.calculate_artifacts(img)
            result['oversharp'] = artifacts['oversharp']
            result['ringing'] = artifacts['ringing']

            # Self-consistency (if original images directory exists)
            if self.original_images_dir is not None:
                # Try to find original image
                # Remove common upscaling suffixes from filename
                original_name = image_path.name

                # Remove suffixes like _bicubic_2x, _realesrgan_4x, etc.
                import re
                # Pattern: _methodname_Nx before extension
                original_name = re.sub(r'_[a-zA-Z0-9]+_\d+x\.(jpg|jpeg|png|bmp|webp|tiff|tif)$',
                                      r'.\1', original_name, flags=re.IGNORECASE)

                original_path = self.original_images_dir / original_name

                if original_path.exists():
                    img_original = self._load_image(original_path)
                    if img_original is not None:
                        result['self_consistency'] = self.calculate_self_consistency(
                            img_original, img
                        )
                    else:
                        result['self_consistency'] = None
                else:
                    result['self_consistency'] = None
            else:
                result['self_consistency'] = None

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def evaluate_batch(self, image_paths: List[Path], verbose: bool = True,
                      n_workers: int = 4) -> List[Dict]:
        """Evaluate quality for multiple images in parallel.

        Args:
            image_paths: List of paths to image files.
            verbose: If True, shows progress bar.
            n_workers: Number of parallel workers (default: 4).

        Returns:
            List of dictionaries containing quality scores.
        """
        if verbose:
            print(f"\nEvaluating quality for {len(image_paths)} images (using {n_workers} workers)...")

        # Use multiprocessing for parallel evaluation
        results = [None] * len(image_paths)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_evaluate_single_image, img_path, self.original_images_dir): idx
                for idx, img_path in enumerate(image_paths)
            }

            # Process results as they complete
            if verbose:
                with tqdm(total=len(image_paths), desc="Evaluating quality") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            results[idx] = {
                                'image_path': str(image_paths[idx]),
                                'image_name': image_paths[idx].name,
                                'status': 'error',
                                'error': str(e)
                            }
                        pbar.update(1)
            else:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {
                            'image_path': str(image_paths[idx]),
                            'image_name': image_paths[idx].name,
                            'status': 'error',
                            'error': str(e)
                        }

        if verbose:
            successful = sum(1 for r in results if r['status'] == 'success')
            print(f"✓ Complete: {successful}/{len(results)} successful")

        return results

    def compare_methods(self, base_dir: Union[str, Path], verbose: bool = True,
                       n_workers: int = 4) -> Dict:
        """Compare quality across different upscaling methods.

        Args:
            base_dir: Base directory containing subdirectories for each method.
            verbose: If True, shows progress information.
            n_workers: Number of parallel workers (default: 4).

        Returns:
            Dictionary with quality comparison results.
        """
        base_dir = Path(base_dir)

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")

        # Find all method subdirectories
        method_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        if verbose:
            print(f"\nFound {len(method_dirs)} upscaling methods to compare")

        comparison = {}

        for method_dir in method_dirs:
            method_name = method_dir.name

            if verbose:
                print(f"\nEvaluating: {method_name}")

            # Get all images in this method directory
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
            image_paths = []
            for ext in valid_extensions:
                image_paths.extend(method_dir.glob(f"*{ext}"))
                image_paths.extend(method_dir.glob(f"*{ext.upper()}"))

            # Filter out hidden files
            image_paths = [p for p in image_paths if not p.name.startswith('.')]

            if not image_paths:
                if verbose:
                    print(f"  No images found in {method_name}")
                continue

            # Evaluate all images for this method
            results = self.evaluate_batch(image_paths, verbose=verbose, n_workers=n_workers)

            # Calculate aggregate statistics
            sharpness_scores = [r['sharpness'] for r in results if r.get('sharpness') is not None]
            edge_scores = [r['edge_strength'] for r in results if r.get('edge_strength') is not None]
            contrast_scores = [r['contrast'] for r in results if r.get('contrast') is not None]
            oversharp_scores = [r['oversharp'] for r in results if r.get('oversharp') is not None]
            ringing_scores = [r['ringing'] for r in results if r.get('ringing') is not None]
            consistency_scores = [r['self_consistency'] for r in results
                                 if r.get('self_consistency') is not None]

            comparison[method_name] = {
                'num_images': len(results),
                'sharpness_mean': np.mean(sharpness_scores) if sharpness_scores else None,
                'sharpness_std': np.std(sharpness_scores) if sharpness_scores else None,
                'edge_strength_mean': np.mean(edge_scores) if edge_scores else None,
                'edge_strength_std': np.std(edge_scores) if edge_scores else None,
                'contrast_mean': np.mean(contrast_scores) if contrast_scores else None,
                'contrast_std': np.std(contrast_scores) if contrast_scores else None,
                'oversharp_mean': np.mean(oversharp_scores) if oversharp_scores else None,
                'oversharp_std': np.std(oversharp_scores) if oversharp_scores else None,
                'ringing_mean': np.mean(ringing_scores) if ringing_scores else None,
                'ringing_std': np.std(ringing_scores) if ringing_scores else None,
                'self_consistency_mean': np.mean(consistency_scores) if consistency_scores else None,
                'self_consistency_std': np.std(consistency_scores) if consistency_scores else None,
                'results': results
            }

        return comparison
