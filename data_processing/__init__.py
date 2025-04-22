"""
Data Processing for MoE-GAN MS-COCO Dataset

This package contains the data processing pipeline for the MoE-GAN model.
"""

# Import key classes directly to make them available from the package
from .data_processing_pipeline import ProcessedMSCOCODataset

__all__ = ['ProcessedMSCOCODataset']