import numpy as np
import h5py

import os,sys
import psutil
import pathlib

import json
import pandas as pd


from timeit import default_timer as timer





def print_proc_info():
    global process 
    print('**Memory: {} Mo'.format(process.memory_info().rss/1024/1024))

def dot_aligned(seq):
    """
    format float to string aligned on the dot
    """
    snums = [str(n) for n in seq]
    dots = [len(s.split('.', 1)[0]) for s in snums]
    m = max(dots)
    return [' '*(m - d) + s for s, d in zip(snums, dots)]

def compute_distance(coor, mode='great_circle'):
    """
    Compute the distance matrix between a list of (lat, lon)
    using the custom distance of geopy on sphere

    Parameter
    ----------
    coor : list of (lat, lon) pair
   
    Return
    ------
    array of distance in km

    Example
    -------
    >>> coor = [(51.5073219,  -0.1276474), (43.60474961791462, 1.4433318113602096), (43.48648160219367, 2.3760056748312457)]
    >>> pdist(coor, lambda u,v: distance(u,v).km)
    >>> array([886.48233373, 911.34709254,  76.51107868])

    """
    from geopy.distance import distance, great_circle
    from scipy.spatial.distance import pdist
    
    t0 = timer()

    # slower but more precise
    if mode=='precise':
        d = pdist(coor, lambda u,v: distance(u,v).km)
    
        print('t0', timer()-t0)
        t0 = timer()

    # faster but less than 1% of error
    if mode=='great_circle':
        d = pdist(coor, lambda u,v: great_circle(u,v).km)
    
        print('t1', timer()-t0)
        t0 = timer()

    return d
    
def compute_distance2(coor):
    """
    numpy version of the geopy 'great_circle' distance algorithm
    
    Notes
    -----
    30x faster with 200pts
    10x faster with 725pts
    Be careful to the combinations that may grow faster and slow down
    the process for bigger number.
    """
    import itertools as it
   
    combin = np.array(list(it.combinations(range(coor.shape[0]), 2)))

    coor = np.radians(coor)[combin]

    #print(coor.shape)

    lat1 = coor[:,0,0]
    lng1 = coor[:,0,1]
    lat2 = coor[:,1,0]
    lng2 = coor[:,1,1]

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    EARTH_RADIUS = 6371.009
    #EARTH_RADIUS = 1./np.pi # debug
    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                   (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
              sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return d*EARTH_RADIUS


def read_lowlevelAPI_h5py(fname, dname, pts, resol):
    '''
    Use h5py low-API to read hyperslab
    https://support.hdfgroup.org/ftp/HDF5/examples/python/hdf5examples-py/low_level/h5ex_d_hyper.py
    '''

    if resol=='4km': off = 2 # temporary for test, should be 0
    elif resol=='1km': off = 2
    elif resol=='300m': off = 6

    pts -= off # remove offset to have the top left corner of the hyperslab
    print('pts.shape:', pts.shape)

    fname = fname.encode()
    dname = dname.encode()

    ## Open the file.        
    file_id = h5py.h5f.open(fname, h5py.h5f.ACC_RDONLY)
    dset_id = h5py.h5d.open(file_id, dname)

    ## First create the first hyperslab with H5S_SELECT_SET_F
    filespace = dset_id.get_space()
    start = (0, *pts[0])
    count = (1, 2*off, 2*off)
    filespace.select_hyperslab(start, count, op=h5py.h5s.SELECT_SET)

    ## Then loop on all the others and add them to the selection with H5S_SELECT_OR_F
    for start in pts[1:]:
        filespace.select_hyperslab((0,*start), count, op=h5py.h5s.SELECT_OR)


    print('selected points nb:', filespace.get_select_npoints())
    print('hyperslab block nb:', filespace.get_select_hyper_nblocks())
    #print(filespace.get_select_hyper_blocklist())

    ## Read the dataset with the previous hyperslab
    ## data type in h5 file: H5T_STD_U16LE
    
    if 1:
        res_size = (pts.shape[0], 2*off, 2*off)
        mspace = h5py.h5s.create_simple(res_size)
        read_data = np.zeros(res_size, dtype=np.uint16)
        dset_id.read(mspace, filespace, read_data)
    else:
        # read simply all : ok
        read_data = np.zeros((1, 4200, 10800), dtype=np.uint16)
        #dset_id.read(h5py.h5s.ALL, h5py.h5s.ALL, read_data)
        dset_id.read(h5py.h5s.ALL, filespace, read_data)


    filespace.close()
    mspace.close()
    dset_id.close()
    file_id.close()

    #print(read_data)


def coor_to_index(coor, resol, fmt='slice'):
    """
    Parameters
    ----------

    coor : list of (lat, lon) pair in decimal degree

    resol : string among '4km', '1km', '300m'

    fmt : string among 'slice', 'hyperslab'

    Note
    ----

    c3s data:
    - shape is (1, lat_len, lon_len)
    - lat extend from -60 to 80 deg
    - lon extend from -180 to 180 deg
    
    Here we want to extract:
    - 1x1 px for 4km
    - 4x4 px for 1km
    - 12x12 px for 300m
    """
    ## Get the param corresponding to the resolution
    param = {'4km':(4200, 10800, 0), '1km':(15680, 40320, 2), '300m':('xxx', 'xxx', 6)}
    lat_len, lon_len, bbox_off = param[resol]
    ## Convert lat/lon to array indices
    ilat = np.rint( lat_len * (1-(coor.T[0]+ 60)/140) ) # data from -60 to +80 deg lat
    ilon = np.rint( lon_len *    (coor.T[1]+180)/360  ) # data from -180 to 180 deg lon
    
    res_idx = np.array((ilat,ilon)).T.astype(int)

    print("global:", lat_len, lon_len, "reduced:", int(ilat.max()-ilat.min()), int(ilon.max()-ilon.min()))

    ## Make a bounding box around the coordinate
    if fmt=='slice':
        if bbox_off==0: 
            res_slice = [(0, ilat, ilon) for (ilat,ilon) in res_idx]
        else:
            res_slice = [(0, slice(ilat-bbox_off, ilat+bbox_off), slice(ilon-bbox_off, ilon+bbox_off)) for (ilat,ilon) in res_idx]
        return res_slice
    else:
        return res_idx



def test_toulouse(fpath):
    """
    Simple test to read c3s data and print indices of a bounding box around Toulouse
    """
    with h5py.File(fpath, 'r') as h5f:

        print_proc_info()

        lon = h5f['lon'][:]
        lat = h5f['lat'][:]

        ## Toulouse
        coor = (43.60472282375283, 1.443320149559777)

        ## Convert lat/lon to array indices
        ilat = int( lat.shape[0] * (1-(coor[0]+60)/140) ) # data from -60 to +80 deg lat
        ilon = int( lon.shape[0] *   (coor[1]+180)/360 ) # data from -180 to 180 deg lon
        print(ilat, ilon)

        ## Get a bounding box around the coordinate
        s = 10
        bbox = (0, slice(ilat-s,ilat+s), slice(ilon-s,ilon+s))
        print('bounding box:', bbox)
        data = h5f['LAI'][bbox]

    print('lat:', lat.shape, lat.min(), lat.max())
    print('lon:', lon.shape, lon.min(), lon.max())

    # diff test on lat/lon data
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    print(dlon.min(), dlon.max())
    print(dlat.min(), dlat.max())

    print_proc_info()

    del(data)

    print_proc_info()


def test_supersites(fpath, slices):
    """
    Test to read 100 supersites in a file:
    - M1: read directly each sites on disk with h5py high level api
    - M2: load all data in memory and select afterward in numpy
    - M3: read directly each site on disk with h5py low level api

    4km:
        M1 : selection = ~3.5s
        M2 : load = ~1.5s, selection = 0.015s
        M3 : TODO (point selection not yet implemented)

    1km:
        M1 : selection = ~10s
        M2 : load = ~15s, selection = 0.015s
        M3 : selection = ~2.7s
    """
    with h5py.File(fpath, 'r') as h5f:

        print_proc_info()

        # M1
        if 1: 
            res = np.array([h5f['LAI'][s] for s in slices])
        
        # M2
        else:
            data = h5f['LAI'][:]
            for ii,s in enumerate(slices):
                res = data[s]
    

def load_landval_sites(fpath, nsub):
    df = pd.read_csv(fpath, sep=';', index_col=0)

    #print(df) 
    #print(list(df))

    site_coor = df[['LATITUDE', 'LONGITUDE']].to_numpy()[:nsub]
    #print(site_coor)

    #d0 = compute_distance(site_coor)
    
    t0 = timer()

    d2 = compute_distance2(site_coor)

    print('t2', timer()-t0)
    t0 = timer()

    #print("d0:", d0.min())
    print("d2:", d2.min())

    plot_dist(d2, len(site_coor))

    sys.exit()


def plot_dist(dist, n):
    import matplotlib.pyplot as plt

    img = np.zeros((n,n))

    ## Add threshold
    dist[dist<6] = 1e5
    #dist[dist>=6] /= 1000

    off = 0
    for i in range(n-1):
        img[i,i+1:] = dist[off:off+n-1-i]
        off += n-1-i

    plt.imshow(img)
    plt.savefig('res_dist.png', dpi=800)


def load_supersite_coor(fpath, nsub):
    with open(fpath) as f:
        landval = json.load(f)

    b_in_bbox = 0
    b_no_overlap = 0
    ## Check if lat min and max are in c3s bounding box
    while not (b_in_bbox and b_no_overlap):
        print('random choice...')
        ## Get random subset
        landval_tmp = [landval[i] for i in np.random.choice(len(landval), nsub)]

        ## convert the list of dic into a dic of dic to be more easy to use
        landval_sub = {}
        for ii,i in enumerate(landval_tmp):
            site_name = list(i.keys())[0]
            landval_sub[site_name] = i[site_name]
            #print(ii, site_name)

        landval_names = list(landval_sub.keys())
        #print(len(landval_names))

        ## select only lat/lon
        #site_coor = {k: (landval_sub[k]['coordinates']['lat'], landval_sub[k]['coordinates']['lon']) for k in landval_names}
        site_coor = np.array([[landval_sub[k]['coordinates']['lat'], landval_sub[k]['coordinates']['lon']] for k in landval_names])

        if (site_coor.T[0].min()>-60.) and (site_coor.T[0].max()<80.):
            b_in_bbox = 1
        
        min_dist = compute_distance(site_coor).min() # a bit long, ~5s for 200pts
        print("min distance:", min_dist)
        if min_dist>6: # 4km pixel so round(4*sqrt(2)) used as min between sites
            b_no_overlap = 1

    ## print result
    if 0:
        str_lat = dot_aligned(site_coor.T[0])
        str_lon = dot_aligned(site_coor.T[1])
        for n,lt,ln in zip(landval_names, str_lat, str_lon):
            print("{:16.16} [ {:<15}, {:<15} ]".format(n, lt, ln))

    return site_coor



def main():
    global process 

    root = 'c3s_data'
    root_path = pathlib.Path(root)

    data = {}
    data['4km'] = ['c3s_LAI_19810920000000_GLOBE_AVHRR-NOAA7_V1.0.1.nc','c3s_LAI_19810930000000_GLOBE_AVHRR-NOAA7_V1.0.1.nc']
    data['1km'] = ['c3s_LAI_20200403000000_GLOBE_PROBAV_V2.0.1.nc','c3s_LAI_20200413000000_GLOBE_PROBAV_V2.0.1.nc']

    process = psutil.Process(os.getpid())
    
    ## Load nsub validation sites coordinates
    nsub = 1000
    #site_coor = load_supersite_coor(root_path/'ALBEDOVAL2-database-20150630.json', nsub)
    site_coor = load_landval_sites('LANDVAL2.csv', nsub)
    #site_coor = load_landval_sites('LANDVAL2_short.csv', nsub) # test
    #site_coor = load_landval_sites('LANDVAL2_test.csv', nsub) # test

    ## Convert  to slices
    resol = '1km'
    landval_slice = coor_to_index(site_coor, resol)
    landval_data = coor_to_index(site_coor, resol, fmt='hyperslab')

    #print(np.sort(landval_data, axis=0))

    ## test value
    #landval_data = np.array([[600,600], [601,601]])

    t0 = timer()
    for d in data[resol]:

        fname = (root_path/d).as_posix()

        print_proc_info()

        #test_toulouse(fname)

        ## M3
        read_lowlevelAPI_h5py(fname, 'LAI', landval_data, resol)

        print('t3', timer()-t0)
        t0 = timer()

        ## M1 and M2
        test_supersites(fname, landval_slice)
        
        print('t1', timer()-t0)
        t0 = timer()


    print_proc_info()

if __name__=='__main__':
    main()
