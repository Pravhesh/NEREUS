"""
Species Classification module for NEREUS pipeline
MobileNetV3-based classification of zooplankton species
"""

from .mobilenet_classifier import MobileNetClassifier

__all__ = ['MobileNetClassifier']
