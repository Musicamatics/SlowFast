"""Mock config module"""

class CfgNode(dict):
    """Minimal CfgNode implementation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    
    def merge_from_file(self, *args, **kwargs):
        pass
    
    def merge_from_list(self, *args, **kwargs):
        pass

def get_cfg():
    return CfgNode()

LazyConfig = get_cfg
