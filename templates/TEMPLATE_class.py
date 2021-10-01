import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pathlib

def parse_args():
    """
    Simply parse args given into one continuous string like this:
    key1=value1:key2=value2:etc.
    Return the corresponding dict
    """
    if len(sys.argv)==2:
        kwargs = {i.split('=')[0]:i.split('=')[1] for i in sys.argv[-1].split(':')}
    
        for k,v in kwargs.items():
            print('-- {} : {}'.format(k,v))
        
        return kwargs


class MyClass:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # You can now use self.var1 etc. directly

if __name__=='__main__':

    kwargs = parse_args()

    pyh = MyClass(**kwargs)

    

