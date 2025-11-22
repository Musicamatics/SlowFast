"""Mock detectron2.layers module"""
import torch.nn as nn

class ROIAlign(nn.Module):
    """Mock ROIAlign layer"""
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
    
    def forward(self, input, rois):
        # Simple passthrough for MaskFeat (not used in image classification)
        return input

# Export the class
__all__ = ['ROIAlign']
