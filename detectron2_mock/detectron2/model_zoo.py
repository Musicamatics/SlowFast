"""Mock detectron2.model_zoo module"""

def get_config_file(config_path):
    """Mock get_config_file"""
    return config_path

def get_checkpoint_url(config_path):
    """Mock get_checkpoint_url"""
    return ""

def get(config_path, trained=False):
    """Mock get model"""
    return None

__all__ = ['get_config_file', 'get_checkpoint_url', 'get']
