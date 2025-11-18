"""Feature extraction using pre-trained vision models.

Supports multiple feature extraction backends:
- DINOv2 (facebook/dinov2-small, facebook/dinov2-base)
- ConvNeXtV2 (timm/convnextv2_tiny.fcmae, timm/convnextv2_base.fcmae)

This module provides both low-level extraction (FeatureExtractor) and
high-level pipeline with batch processing (FeatureExtractionPipeline).
"""
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """Unified feature extractor for multiple vision models.

    Attributes:
        model_name (str): Name of the loaded model.
        device_str (str): Device being used for inference.
        model: The loaded model.
        processor: Image processor/preprocessor.
        feature_dim (int): Dimensionality of extracted features.
    """

    SUPPORTED_MODELS = {
        'dinov2-small': 'facebook/dinov2-small',
        'dinov2-base': 'facebook/dinov2-base',
        'convnextv2-tiny': 'timm/convnextv2_tiny.fcmae',
        'convnextv2-base': 'timm/convnextv2_base.fcmae'
    }

    def __init__(self, model_name='dinov2-small', device=None):
        """Initialize the feature extractor.

        Args:
            model_name (str): Model name or alias (e.g., 'dinov2-small').
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.device_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)

        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            full_model_name = self.SUPPORTED_MODELS[model_name]
        else:
            full_model_name = model_name

        # Load model based on type
        if 'dinov2' in model_name.lower():
            self._load_dinov2(full_model_name)
        elif 'convnext' in model_name.lower():
            self._load_convnext(full_model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.eval()

    def _load_dinov2(self, model_name):
        """Load DINOv2 model from HuggingFace."""
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # DINOv2 feature dimensions
        if 'small' in model_name:
            self.feature_dim = 384
        elif 'base' in model_name:
            self.feature_dim = 768
        elif 'large' in model_name:
            self.feature_dim = 1024
        else:
            self.feature_dim = 768  # default

    def _load_convnext(self, model_name):
        """Load ConvNeXtV2 model from timm."""
        # Extract model name from full path
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name

        # Load model using timm
        self.model = timm.create_model(
            model_short_name,
            pretrained=True,
            num_classes=0  # Remove classification head to get features
        ).to(self.device)

        # Get data config for preprocessing
        data_config = timm.data.resolve_model_data_config(self.model)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        # ConvNeXtV2 feature dimensions
        self.feature_dim = self.model.num_features

    def extract_from_path(self, image_path):
        """Extract features from an image file.

        Args:
            image_path (str or Path): Path to the image file.

        Returns:
            np.ndarray: Feature vector of shape (feature_dim,).
        """
        image = Image.open(image_path).convert('RGB')
        return self.extract_from_image(image)

    def extract_from_image(self, image):
        """Extract features from a PIL Image.

        Args:
            image (PIL.Image.Image): PIL Image object.

        Returns:
            np.ndarray: Feature vector of shape (feature_dim,).
        """
        # Preprocess image
        if 'dinov2' in self.model_name.lower():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:  # ConvNeXt
            inputs = self.processor(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            if 'dinov2' in self.model_name.lower():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:  # ConvNeXt
                features = self.model(inputs).cpu().numpy()

        return features.flatten()

    def extract_batch(self, images):
        """Extract features from multiple images in batch.

        Args:
            images (list): List of images. Each can be PIL.Image or path string.

        Returns:
            np.ndarray: Feature matrix of shape (n_images, feature_dim).
        """
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)

        # Preprocess batch
        if 'dinov2' in self.model_name.lower():
            inputs = self.processor(images=loaded_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:  # ConvNeXt
            inputs = torch.stack([self.processor(img) for img in loaded_images]).to(self.device)

        # Extract features
        with torch.no_grad():
            if 'dinov2' in self.model_name.lower():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:  # ConvNeXt
                features = self.model(inputs).cpu().numpy()

        return features


class FeatureExtractionPipeline:
    """High-level pipeline for extracting features from image datasets.

    This class provides convenient methods for batch processing, progress tracking,
    and saving/loading features.

    Attributes:
        extractor (FeatureExtractor): Feature extractor instance.
        feature_dim (int): Dimensionality of features.
        model_name (str): Name of the model being used.
    """

    def __init__(self, model_name='dinov2-small', device=None):
        """Initialize feature extraction pipeline.

        Args:
            model_name (str): Model name for feature extraction.
            device (str, optional): Device to use ('cuda', 'cpu', or None).
        """
        self.extractor = FeatureExtractor(model_name, device)
        self.feature_dim = self.extractor.feature_dim
        self.model_name = model_name

    def extract_from_paths(self, image_paths, batch_size=32, show_progress=True):
        """Extract features from a list of image paths.

        Args:
            image_paths (list): List of image file paths.
            batch_size (int): Batch size for processing.
            show_progress (bool): Show progress bar.

        Returns:
            np.ndarray: Feature matrix of shape (n_images, feature_dim).
        """
        features_list = []
        iterator = range(0, len(image_paths), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting features ({self.model_name})")

        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]

            try:
                batch_features = self.extractor.extract_batch(batch_paths)
                features_list.append(batch_features)
            except Exception as e:
                # If batch fails, process individually
                if show_progress:
                    print(f"\nBatch failed, processing individually: {e}")

                for path in batch_paths:
                    try:
                        feat = self.extractor.extract_from_path(path)
                        features_list.append(feat.reshape(1, -1))
                    except Exception as e2:
                        if show_progress:
                            print(f"  Failed {Path(path).name}: {e2}")
                        # Append zero vector for failed images
                        features_list.append(np.zeros((1, self.feature_dim)))

        return np.vstack(features_list)

    def extract_and_save(self, image_paths, output_file, batch_size=32, metadata=None):
        """Extract features and save to file.

        Args:
            image_paths (list): List of image paths.
            output_file (str or Path): Output file path (.npz).
            batch_size (int): Batch size for processing.
            metadata (dict, optional): Additional metadata to save.

        Returns:
            np.ndarray: Extracted features.
        """
        features = self.extract_from_paths(image_paths, batch_size)

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        save_dict = {
            'features': features,
            'image_paths': image_paths,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim
        }

        if metadata:
            save_dict.update(metadata)

        np.savez(output_file, **save_dict)
        print(f"✓ Features saved: {output_file}")

        return features

    @staticmethod
    def load_features(feature_file):
        """Load features from file.

        Args:
            feature_file (str or Path): Path to .npz file.

        Returns:
            dict: Dictionary containing features and metadata.
        """
        data = np.load(feature_file, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def __repr__(self):
        return f"FeatureExtractionPipeline(model={self.model_name}, dim={self.feature_dim})"
