import h5py
import tools


class Pyh5dump:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        with h5py.File(self.fname, 'r') as h:
            print(h[self.var][int(self.s):int(self.e)])


if __name__=='__main__':

    kwargs = tools.parse_args()

    pyh = Pyh5dump(**kwargs)

    print(pyh.fname)
    

