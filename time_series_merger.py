import pandas as pd
import h5py
import numpy as np
import pathlib
import tools
import pandas as pd
import sys


class TimeSeriesMerger():
    """
    Merge two cache files f1 and f2.
    f1 is overwritten by f2 where global_id's overlap.
    """

    def __init__(self, f1, f2, ):

        f1 = pathlib.Path(f1)
        f2 = pathlib.Path(f2)

        # get the product from initial cache file path or from attribute of merged time series
        v1 = self._get_product(f1)
        v2 = self._get_product(f2)
        assert v1==v2, ('Not the same product:', v1, v2)
        self.product = v1

        self.f1 = f1
        self.f2 = f2
        self.output_path = pathlib.Path('./output_merge')
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _get_product(self, path):
        """
        Get the product name.

        Use the path if it's an initial cache file or the attribute if it's
        a already merged time series.

        Return product name
        """
        if path.parts[0]=='output_extract':
            return '_'.join(path.parts[1].split('_')[:-1])
        elif path.parts[0]=='output_merge':
            with h5py.File(path, mode='r') as h:
                return h.attrs['product']

    def _get_file_vars(self, fname):
        """
        Read and sort variables from cache file
        
        Return 2 dict
        """
        with h5py.File(fname, mode='r') as h:
            dic_meta = {'ts_dates':None, 'global_id':None, 'point_names':None}
            dic_var = {}
            for k in h.keys():
                # keep var without '_d?' suffix
                if k[-3:-1] != '_d':
                    if k in dic_meta.keys():
                        dic_meta[k] = h[k][:]
                    else:
                        dic_var[k] = h[k][:]

        dic_meta, dic_var = self._trim_timeseries(dic_meta, dic_var)
        
        return dic_meta, dic_var

    def _trim_timeseries(self, meta, var):
        """ 
        Trim data in time to remove empty start and end.
        """
        ## Get indices to trim based on ts_dates=0
        i = 0
        for d in meta['ts_dates']:
            if d==0:
                i+=1
            else:
                break
        j = 0
        for d in meta['ts_dates'][::-1]:
            if d==0:
                j+=1
            else:
                break

        print('INFO: remove {} rows at the begining and {} at the end.'.format(i,j))
        
        for v in ['global_id', 'ts_dates']:
            meta[v] = meta[v][i:len(meta[v])-j]
        for v in var.keys():
            var[v] = var[v][i:len(var[v])-j]

        return meta, var


    def merge(self):
        
        ## Read both dataset 
        meta1, var1 = self._get_file_vars(self.f1)
        meta2, var2 = self._get_file_vars(self.f2)
        
        ## suppose for now that all cache files have the same variables and landval sites
        assert np.array_equal(meta1['point_names'], meta2['point_names']), 'LANDVAL sites are not equals'
        assert np.array_equal(np.array(var1.keys()), np.array(var2.keys())), 'Channels are not equals'

        ## Compute new dimension
        g1 = meta1['global_id']
        g2 = meta2['global_id']
        
        print("g1.shape,g2.shape=", g1.shape,g2.shape)
        print("g1[0],g1[-1]=", g1[0],g1[-1])
        print("g2[0],g2[-1]=", g2[0],g2[-1])

        if g1[0]<g2[0]:
            dmin = g1[0]
        else:
            dmin = g2[0]
        if g1[-1]>g2[-1]:
            dmax = g1[-1]
        else:
            dmax = g2[-1]

        len_new = dmax-dmin+1
        print("dmin,dmax=", dmin,dmax)
        print("new_len=", len_new)

        s1 = slice(g1[0]-dmin, g1[-1]-dmin+1)
        s2 = slice(g2[0]-dmin, g2[-1]-dmin+1)
        

        ## Merge meta data
        var_new = {}
        meta_new = {}
        meta_new['point_names'] = meta1['point_names']
        for vm in ['global_id', 'ts_dates']:
            meta_new[vm] = np.zeros((len_new,), dtype=meta1[vm].dtype)
            meta_new[vm][s1] = meta1[vm]
            meta_new[vm][s2] = meta2[vm]
    
        ## Merge data
        for v in var1.keys():
            var_new[v] = np.zeros((len_new, 1, len(meta1['point_names'])), dtype=var1[v].dtype)
            var_new[v][s1] = var1[v]
            var_new[v][s2] = var2[v]
    

        ## Save file
        fname = 'merged_timeseries_{}_{}{:02d}_{}{:02d}.h5'.format(self.product.replace('_',''), 1970+dmin//36, dmin%36, 1970+dmax//36, dmax%36)
        outpath = str(self.output_path / fname)
        with h5py.File(outpath, mode='w') as h:
            for k,v in meta_new.items():
                h[k] = v
            for k,v in var_new.items():
                h[k] = v
            h.attrs['product'] = self.product
        print('Merged file saved to:', outpath)


if __name__=='__main__':
    
    kwargs = tools.parse_args()

    tsm = TimeSeriesMerger(**kwargs)
    
    tsm.merge()


    
