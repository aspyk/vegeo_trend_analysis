import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pathlib
import psutil
import time
import datetime as dt

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

    else:
        return {}


class MemoryMonitor:
    """
    Usage:
    python TOOL_mem_monitor.py [t=<time>]:[r=<rate>]

    With:
        -t : the tool will monitor the memory for <time> seconds.
        -r : the tool will probe the memory each <rate> seconds.

    Example:
    python TOOL_mem_monitor.py t=20:r=0.1
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # You can now use self.var1 etc. directly

        self.log = []
        if hasattr(self, 't'):
            self.t = float(self.t)
        else:
            self.t = 20.

        if hasattr(self, 'r'):
            self.r = float(self.r)
        else:
            self.r = 0.1

        self.n = int(self.t/self.r)

        t0 = dt.datetime.now()
        print(t0)  

        for i in range(self.n):
            self.log.append(psutil.virtual_memory().percent)
            time.sleep(self.r)
        
        t1 = dt.datetime.now()
        print(t1)
        self.elapsed = t1-t0
        print(self.elapsed)
        print(self.elapsed.total_seconds())

        fname = 'res_mem_monitor_{}.png'.format(t0.strftime('%Y%m%d_%H%M'))
        self.plot(fname=fname)
        print('### Results save to {}.png'.format(fname))

    def plot(self, fname='res_mem_monitor'):

        ## Get list of axes        
        fig, axs = plt.subplots()
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.ravel()

        axs[0].plot(self.elapsed.total_seconds()*np.arange(self.n)/self.n, self.log)

        axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('memory used [%]')
        axs[0].grid()
        #plt.show()
        plt.savefig(fname)

        os.system('feh ' + fname)

if __name__=='__main__':

    kwargs = parse_args()

    mon = MemoryMonitor(**kwargs)

    

