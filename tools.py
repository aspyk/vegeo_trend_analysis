from timeit import default_timer as timer
import sys

class SimpleTimer():
    """
    Usage
    -----
    from tools import SimpleTimer
    ti = SimpleTimer()  # initialize the timer and start counting
        [..some code..]
    ti('label1')        # print the time difference since the last call (here the init)
        [..some code..]
    ti('label2')        # print the time difference since the last call
        [..some code..]
    ti()                # reset the counter without printing anything 
        [..some code..]
    ti('label3')        # print the time difference since the last call
    ti.show()           # print all the previous timing (they are recorded automatically)
    """
    def __init__(self):
        self.t0 = timer()
        self.res = []

    def __call__(self, msg=None):
        if msg is not None:
            dt = timer()-self.t0
            self.res.append([msg, dt])
            print(msg, dt)
        self.t0 = timer()

    def show(self):
        print("** Timing summary **")
        for r in self.res:
            print("{0} {1}".format(*r))

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

