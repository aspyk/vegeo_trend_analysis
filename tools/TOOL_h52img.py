import os,sys
import psutil
import h5py
import numpy as np


def str_to_slice(str_slice):
    """
    Parses a `slice()` from string, like `start:stop:step`.
    """
    res = []
    for value in str_slice.split(','):
        parts = []
        if value:
            parts = value.split(':')
            if len(parts) == 1:
                res.append(int(parts[0]))
                continue
        res.append(slice(*[int(p) if p else None for p in parts]))
    return tuple(res)

def plot(fname, var, idx=None):
    ## Read data
    print('Read data...')
    h5f = h5py.File(fname, 'r')
    v = h5f[var]
    print('initial shape :', v.shape)
    #print('min/max:', np.nanmin(v), '/', np.nanmax(v))

    if v.ndim==1:
        print('ERROR: 1D array. 1D plot not implemented yet')
        sys.exit()

    elif v.ndim==2:
        if idx is None:
            v = v[:]
            print('final shape   :', v.shape)
        else:
            v = v[str_to_slice(idx)]
            print('final shape   :', v.shape)

    elif v.ndim>2:
        if idx is None:
            print('ERROR: slicing need to be provided for dataset with ndim > 2')
            sys.exit()
        else:
            v = v[str_to_slice(idx)]
            print('final shape   :', v.shape)
            if v.ndim>2:
                print('ERROR: sliced array is still ndim>2')
                sys.exit()


        h5f.close()

    ## Print some stats
    print('min/max:', np.nanmin(v), '/', np.nanmax(v))

    ## Generate image
    print('Generate image...')

    import matplotlib.pyplot as plt

    ######### custom operation
    #v = np.where(((v>>5)&1)==1, 1, 0) 
    v = (v>>5)&1
    print(v.sum())
    #v = v>>2
    #v[v>3000] = 3000
    #print(np.unique(v))
    #sys.exit()
    #########

    plt.imshow(v, cmap='jet', interpolation='nearest')
    #plt.imshow(v, cmap='tab10')
    imgname = 'h52img_{0}.png'.format(var)
    plt.title('{0}'.format(var))
    plt.colorbar()
    try:
        plt.savefig(imgname)
    except:
        # If permission issue, write image in a default dir
        imgname = '~/'+imgname
        plt.savefig(imgname)


    print('Saved to {}'.format(imgname))

    #os.system('/mnt/lfs/d30/vegeo/fransenr/CODES/tools/TerminalImageViewer/src/main/cpp/tiv ' + imgname)
    os.system('feh ' + imgname)

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1024/1024, 'Mo')


if __name__ == "__main__":
    args = sys.argv[1:]

    ## Check args
    if (len(args) not in [2,3]) or (args[-1]=='-h'):
        print("Usage:")
        print('python h52img.py <path_to_file> <dataset_name> [<slicing>]')
        print('The dataset (origianl or sliced) has to have 2 dimensions.')

        ## Print datasets in file if no dataset has been given
        try:
            h5f = h5py.File(args[-1], 'r')
            print('Hint: datasets in %s :'%args[-1])
            for k in h5f.keys():
                print(k)
        except:
            pass

        sys.exit()

    fname = args[0]
    var = args[1]

    if len(args)==3:
        idx = args[2]
        plot(fname, var, idx)
    else:
        plot(fname, var)



