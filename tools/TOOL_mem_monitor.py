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
    key1=value1,key2=value2,etc.
    Return the corresponding dict
    """
    if len(sys.argv)==2:
        kwargs = {i.split('=')[0]:i.split('=')[1] for i in sys.argv[-1].split(',')}
    
        for k,v in kwargs.items():
            print('-- {} : {}'.format(k,v))
        
        return kwargs

    else:
        return {}


class MemoryMonitor:
    """
    Usage:
    python TOOL_mem_monitor.py [t=<time>],[r=<rate>]

    With:
        -t : the tool will monitor the memory for <time> seconds.
        -r : the tool will probe the memory each <rate> seconds.

    Example:
    python TOOL_mem_monitor.py t=20,r=0.1
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

        print(f'record time: {self.t}s')
        print(f'sample rate: {self.r}s')
        print('Start memory monitoring... (CTRL+C to break the loop)')

        t0 = dt.datetime.now()
        print(f'START: {t0}')  

        self.mem_total = psutil.virtual_memory().total

        ## Add a try block to be able to break the for loop with ctrl+C
        try:
            for i in range(self.n):
                # .percent = (.total - .available) / .total * 100
                #self.log.append(psutil.virtual_memory().percent)
                self.log.append(self.mem_total-psutil.virtual_memory().available)
                time.sleep(self.r)
        except KeyboardInterrupt:
            pass

        t1 = dt.datetime.now()
        print(f'STOP: {t1}')  
        self.elapsed = t1-t0
        #print(self.elapsed.total_seconds())
        print(f'Monitoring duration: {self.elapsed}')
        
        self.log = np.array(self.log)

        fname = 'res_mem_monitor_{}.png'.format(t0.strftime('%Y%m%d_%H%M'))
        self.plot(fname=fname)
        print('### Results save to {}'.format(fname))

    def get_bytes_unit(self, size):
        # 2**10 = 1024
        power = 2**10
        n = 0
        power_labels = {0 : '', 1: 'k', 2: 'M', 3: 'G', 4: 'T'}
        while size > power:
            size /= power
            n += 1
        return n, power_labels[n]

    def plot(self, fname='res_mem_monitor'):

        ## Get list of axes        
        fig, axs = plt.subplots()
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.ravel()
        
        time = (self.elapsed.total_seconds()*np.arange(self.n)/self.n)[:len(self.log)]

        axs[0].plot(time, 100*self.log/self.mem_total)

        # first axis in percent, second in bytes
        ax2 = axs[0].twinx()
        mn, mx = axs[0].get_ylim()
        power, unit = self.get_bytes_unit(0.01*mx*self.mem_total)
        ax2.set_ylim(0.01*mn*self.mem_total/1024**power, 0.01*mx*self.mem_total/1024**power)

        axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('[%]')
        ax2.set_ylabel(f'[{unit}bytes]')
        axs[0].grid()
        axs[0].set_title(f'Memory usage - Total {psutil._common.bytes2human(self.mem_total)}')

        #plt.show()
        plt.savefig(fname, bbox_inches='tight')

if __name__=='__main__':

    kwargs = parse_args()

    mon = MemoryMonitor(**kwargs)

    

