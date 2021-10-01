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


class Toc:
    def __init__(self, **kwargs):
        for key in kwargs:
            self.from_lvl = 2
            self.to_lvl = 3
            setattr(self, key, kwargs[key])

        # You can now use self.var1 etc. directly

    def create(self):
        with open(self.input, 'r') as f:
            lines = f.readlines()

        toc = []
        anchors = []
        for idl,l in enumerate(lines):
            if l.startswith('##'):
                print(l)
                depth = l.count('#')
                if self.from_lvl <= depth <= self.to_lvl:
                    title = l.replace('#','').strip()
                    tag = title.replace(' ','').lower()
                    anchor = '<a name="{}"></a>'.format(tag)
                    tocline = "{}1. [{}](#{})".format(' '*4*(depth-2), title, tag)
                    toc.append(tocline)
                    anchors.append([idl, anchor])


        for a in anchors[::-1]:
            lines.insert(a[0], a[1]+'\n')

        with open('test.md', 'w') as f:
            f.writelines(lines)
        

        print('')
        print('## Table of contents')
        for l in toc:
            print(l)

if __name__=='__main__':

    kwargs = parse_args()

    toc = Toc(**kwargs)

    toc.create()

    

