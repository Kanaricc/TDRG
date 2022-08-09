from typing import Any, List, Union

import torch
from torch import Tensor
from chinopie.datasets.coco2014 import COCO2014Dataset
from loguru import logger
import copy

class COCO2014Partial(COCO2014Dataset):
    def __init__(self, root: str, preprocess: Any, phase: str = "train", negatives_as_neg1=False):
        super().__init__(root, preprocess, phase,negatives_as_neg1)
        self.origin_img_lists=copy.deepcopy(self.img_list)
    
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
    
    def get_full_labels(self,raw_ids:List[int]):
        labels=[]
        for id in raw_ids:
            t=torch.zeros(self.num_classes,dtype=torch.int)
            t.fill_(-1)
            t[self.origin_img_lists[id]['labels']]=1
            labels.append(t.unsqueeze(0))
        
        return torch.cat(labels)