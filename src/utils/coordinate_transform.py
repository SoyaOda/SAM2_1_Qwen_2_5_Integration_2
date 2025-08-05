"""
Coordinate transformation utilities for LISA-Kai model
Handles coordinate mapping between different resolution spaces (Qwen 448x448 <-> SAM 1024x1024)
"""
import torch
from typing import Tuple, List, Optional, Union
import numpy as np


class CoordinateTransform:
    """
    Handles coordinate transformations between different resolution spaces
    
    This class manages the mapping between:
    - Original image coordinates
    - Qwen model space (448x448 with aspect ratio preserving padding)
    - SAM model space (1024x1024 with longest side resizing)
    """
    
    def __init__(
        self, 
        orig_size: Tuple[int, int],  # (H, W)
        qwen_size: int = 448,
        sam_size: int = 1024
    ):
        """
        Initialize coordinate transformer
        
        Args:
            orig_size: Original image size as (height, width)
            qwen_size: Qwen model input size (default: 448)
            sam_size: SAM model input size (default: 1024)
        """
        self.orig_h, self.orig_w = orig_size
        self.qwen_size = qwen_size
        self.sam_size = sam_size
        
        # Calculate Qwen transformation parameters (aspect ratio preserving)
        self.scale_qwen = qwen_size / max(self.orig_h, self.orig_w)
        self.scaled_h_qwen = int(self.orig_h * self.scale_qwen)
        self.scaled_w_qwen = int(self.orig_w * self.scale_qwen)
        
        # Calculate padding for Qwen (centered)
        self.pad_top_qwen = (qwen_size - self.scaled_h_qwen) // 2
        self.pad_left_qwen = (qwen_size - self.scaled_w_qwen) // 2
        
        # Calculate SAM transformation parameters (longest side to 1024)
        self.scale_sam = sam_size / max(self.orig_h, self.orig_w)
        self.scaled_h_sam = int(self.orig_h * self.scale_sam)
        self.scaled_w_sam = int(self.orig_w * self.scale_sam)
        
        # Calculate padding for SAM (top-left aligned with black padding)
        self.pad_top_sam = 0  # SAM typically uses top-left alignment
        self.pad_left_sam = 0
    
    def qwen_to_original(
        self, 
        x_qwen: Union[float, torch.Tensor], 
        y_qwen: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from Qwen space (448x448) to original image space
        
        Args:
            x_qwen: X coordinate(s) in Qwen space
            y_qwen: Y coordinate(s) in Qwen space
            
        Returns:
            Tuple of (x_orig, y_orig) in original image space
        """
        x_orig = (x_qwen - self.pad_left_qwen) / self.scale_qwen
        y_orig = (y_qwen - self.pad_top_qwen) / self.scale_qwen
        
        return x_orig, y_orig
    
    def original_to_qwen(
        self, 
        x_orig: Union[float, torch.Tensor], 
        y_orig: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from original image space to Qwen space (448x448)
        
        Args:
            x_orig: X coordinate(s) in original space
            y_orig: Y coordinate(s) in original space
            
        Returns:
            Tuple of (x_qwen, y_qwen) in Qwen space
        """
        x_qwen = x_orig * self.scale_qwen + self.pad_left_qwen
        y_qwen = y_orig * self.scale_qwen + self.pad_top_qwen
        
        return x_qwen, y_qwen
    
    def sam_to_original(
        self, 
        x_sam: Union[float, torch.Tensor], 
        y_sam: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from SAM space (1024x1024) to original image space
        
        Args:
            x_sam: X coordinate(s) in SAM space
            y_sam: Y coordinate(s) in SAM space
            
        Returns:
            Tuple of (x_orig, y_orig) in original image space
        """
        x_orig = (x_sam - self.pad_left_sam) / self.scale_sam
        y_orig = (y_sam - self.pad_top_sam) / self.scale_sam
        
        return x_orig, y_orig
    
    def original_to_sam(
        self, 
        x_orig: Union[float, torch.Tensor], 
        y_orig: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from original image space to SAM space (1024x1024)
        
        Args:
            x_orig: X coordinate(s) in original space
            y_orig: Y coordinate(s) in original space
            
        Returns:
            Tuple of (x_sam, y_sam) in SAM space
        """
        x_sam = x_orig * self.scale_sam + self.pad_left_sam
        y_sam = y_orig * self.scale_sam + self.pad_top_sam
        
        return x_sam, y_sam
    
    def qwen_to_sam(
        self, 
        x_qwen: Union[float, torch.Tensor], 
        y_qwen: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from Qwen space (448x448) to SAM space (1024x1024)
        
        Args:
            x_qwen: X coordinate(s) in Qwen space
            y_qwen: Y coordinate(s) in Qwen space
            
        Returns:
            Tuple of (x_sam, y_sam) in SAM space
        """
        # First convert to original space
        x_orig, y_orig = self.qwen_to_original(x_qwen, y_qwen)
        
        # Then convert to SAM space
        x_sam, y_sam = self.original_to_sam(x_orig, y_orig)
        
        return x_sam, y_sam
    
    def sam_to_qwen(
        self, 
        x_sam: Union[float, torch.Tensor], 
        y_sam: Union[float, torch.Tensor]
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """
        Convert coordinates from SAM space (1024x1024) to Qwen space (448x448)
        
        Args:
            x_sam: X coordinate(s) in SAM space
            y_sam: Y coordinate(s) in SAM space
            
        Returns:
            Tuple of (x_qwen, y_qwen) in Qwen space
        """
        # First convert to original space
        x_orig, y_orig = self.sam_to_original(x_sam, y_sam)
        
        # Then convert to Qwen space
        x_qwen, y_qwen = self.original_to_qwen(x_orig, y_orig)
        
        return x_qwen, y_qwen
    
    def transform_bbox_qwen_to_sam(
        self, 
        bbox_qwen: Union[List[float], torch.Tensor]
    ) -> Union[List[float], torch.Tensor]:
        """
        Transform bounding box from Qwen space to SAM space
        
        Args:
            bbox_qwen: Bounding box in Qwen space as [x1, y1, x2, y2]
            
        Returns:
            Bounding box in SAM space as [x1, y1, x2, y2]
        """
        if isinstance(bbox_qwen, torch.Tensor):
            x1, y1, x2, y2 = bbox_qwen
            x1_sam, y1_sam = self.qwen_to_sam(x1, y1)
            x2_sam, y2_sam = self.qwen_to_sam(x2, y2)
            return torch.stack([x1_sam, y1_sam, x2_sam, y2_sam])
        else:
            x1, y1, x2, y2 = bbox_qwen
            x1_sam, y1_sam = self.qwen_to_sam(x1, y1)
            x2_sam, y2_sam = self.qwen_to_sam(x2, y2)
            return [x1_sam, y1_sam, x2_sam, y2_sam]
    
    def get_valid_region_qwen(self) -> Tuple[int, int, int, int]:
        """
        Get the valid (non-padded) region in Qwen space
        
        Returns:
            Tuple of (top, left, bottom, right) for valid region
        """
        top = self.pad_top_qwen
        left = self.pad_left_qwen
        bottom = top + self.scaled_h_qwen
        right = left + self.scaled_w_qwen
        
        return top, left, bottom, right
    
    def get_valid_region_sam(self) -> Tuple[int, int, int, int]:
        """
        Get the valid (non-padded) region in SAM space
        
        Returns:
            Tuple of (top, left, bottom, right) for valid region
        """
        top = self.pad_top_sam
        left = self.pad_left_sam
        bottom = top + self.scaled_h_sam
        right = left + self.scaled_w_sam
        
        return top, left, bottom, right