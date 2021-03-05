import pandas as pd
import h5py
import numpy as np
import pathlib
import tools
import sys
import itertools
import generic


class TimeSeriesMerger():
    """
    Merge several cache files into one. The last file overwrite the previous ones:
    If we have <arg1> <arg2> <arg3> ...
    arg1 is overwritten by arg2, then arg2 and arg1 by arg3 etc..
    """

    def __init__(self, fname_list, prod_list=None):
        """
        fname_list: list of cache files to merge

        prod_list: l
        """

        files =  [pathlib.Path(i) for i in fname_list]

        # get the product from initial cache file path or from attribute of merged time series
        prod = [self._get_product(f) for f in files]
        assert len(set(prod)) <= 1, ('Not the same product in:', prod)
        self.product = prod[0]

        print("Product to merge:", self.product)

        self.files = files
        self.output_path = pathlib.Path('./output_extract') / (self.product+'_MERGED')
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
            dic_meta = {}
            dic_var = {}
            for k in h['meta'].keys():
                dic_meta[k] = h['meta/'+k][:]
            for k in h['vars'].keys():
                dic_var[k] = h['vars/'+k][:]

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


    def run(self):
        
        ## Read all datasets
        meta_in = []
        var_in = []
        for f in self.files:
            m, v = self._get_file_vars(f)
            meta_in.append(m)
            var_in.append(v)
        
        ## suppose for now that all cache files have the same variables and landval sites
        assert all([np.array_equal(m1['point_names'], m2['point_names']) for m1,m2 in itertools.combinations(meta_in,2)]), 'LANDVAL sites are not equals'
        assert all([np.array_equal(v1.keys(), v2.keys()) for v1,v2 in itertools.combinations(var_in,2)]), 'LANDVAL sites are not equals'
        #assert np.array_equal(meta1['point_names'], meta2['point_names']), 'LANDVAL sites are not equals'
        #assert np.array_equal(np.array(var1.keys()), np.array(var2.keys())), 'Channels are not equals'

        ## Compute new dimension
        gid = [m['global_id'] for m in meta_in]
        
        for g in gid:
            print("gid shape, first, last=", g.shape, g[0],g[-1])

        dmin = min([min(g) for g in gid])
        dmax = max([max(g) for g in gid])

        len_new = dmax-dmin+1
        print("dmin,dmax=", dmin,dmax)
        print("new_len=", len_new)

        loc = [slice(g[0]-dmin, g[-1]-dmin+1) for g in gid]

        ## Merge meta data
        var_new = {}
        meta_new = {}
        meta_new['point_names'] = meta_in[0]['point_names']
        for vm in ['global_id', 'ts_dates']:
            meta_new[vm] = np.zeros((len_new,), dtype=meta_in[0][vm].dtype)
            for l,meta in zip(loc,meta_in):
                meta_new[vm][l] = meta[vm]
    
        ## Merge data
        for v in var_in[0].keys():
            var_new[v] = np.zeros((len_new, 1, len(meta_in[0]['point_names'])), dtype=var_in[0][v].dtype)
            for l,var in zip(loc,var_in):
                var_new[v][l] = var[v]
    

        ## Save file
        fname = 'timeseries_{}{:02d}_{}{:02d}.h5'.format(1970+dmin//36, dmin%36, 1970+dmax//36, dmax%36)
        outpath = self.output_path / fname
        with h5py.File(outpath, mode='w') as h:
            for k,v in meta_new.items():
                h['meta/'+k] = v
            for k,v in var_new.items():
                h['vars/'+k] = v
            h.attrs['product'] = self.product
        print('Merged file saved to:', outpath)

        merged_prod = generic.Product(self.product+'_MERGED',
                                      pd.to_datetime(meta_new['ts_dates'][0], unit='s'),
                                      pd.to_datetime(meta_new['ts_dates'][-1], unit='s'),
                                      'merged')
        return merged_prod


if __name__=='__main__':
    
    #kwargs = tools.parse_args()

    args = sys.argv[1:]

    #tsm = TimeSeriesMerger(**kwargs)
    tsm = TimeSeriesMerger(args)
    
    tsm.run()


    
