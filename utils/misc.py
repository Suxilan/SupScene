import os
import torch

def printf(msg):
    """print only on the main process in distributed training"""
    try:
        local_rank = os.environ.get('LOCAL_RANK', None)
        rank = os.environ.get('RANK', None) 

        if local_rank is not None or rank is not None:
            current_rank = int(local_rank or rank or 0)
            if current_rank == 0:
                print(msg)
        elif torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(msg)
        else:
            print(msg)
    except Exception as e:
        print(f"printf error: {e}, msg: {msg}")