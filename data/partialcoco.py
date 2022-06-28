from typing import Any

import torch
from chinopie.datasets.coco2014 import COCO2014Dataset
from loguru import logger

class COCO2014Partial(COCO2014Dataset):
    def __init__(self, root: str, preprocess: Any, phase: str = "train"):
        super().__init__(root, preprocess, phase)
    
    def drop_labels(self,percent:float,seed):
        tmp=self.get_all_labels()
        torch.manual_seed(seed)
        ran=torch.rand_like(tmp,dtype=torch.float)
        drop_percent=1.0-percent
        ran[ran<drop_percent]=0
        ran[ran>=drop_percent]=1
        ran=ran.int()
        tmp=tmp*ran

        self.apply_new_labels(tmp)
        logger.warning(f"drop to {percent} labels. the hash of new labels is {hash(tmp)}")
