import pandas as pd
import numpy as np
import hashlib
from datetime import *
from pprint import pprint


class Splitter():
    """
    Class to perform recursive partitioning of the selected region into chunks.

    Columns use x index
    Rows use y index

    Args:
        xlim1,xlim2,ylim1,ylim2: global region coordinates on MSG disk.
        parent: use only internally
    """

    def __init__(self, xlim1, xlim2, ylim1, ylim2, parent=None):
        self.input = 'box'

        self.x1 = xlim1
        self.x2 = xlim2
        self.y1 = ylim1
        self.y2 = ylim2

        if parent is None:
            self.parent = self
        else:
            self.parent = parent

        # swap to fit numpy array indexing: array[y,x]
        self.dim = (self.y2-self.y1, self.x2-self.x1)


    def get_limits(self, ref, fmt):
        """
        Return the global limits with the specified format

        Args:
            ref: 'global' or 'local'

            fmt:  - 'tuple' : return tuple
                  - 'str'   : return a string
                  - 'slice' : return slice object
        """
        
        if ref=='global':
            refx = 0
            refy = 0
        elif ref=='local':
            refx = self.parent.x1
            refy = self.parent.y1

        x1 = self.x1-refx
        x2 = self.x2-refx
        y1 = self.y1-refy
        y2 = self.y2-refy

        if fmt=='tuple':
            return (x1, x2, y1, y2)
        elif fmt=='str':
            # Use (x,y) order to print
            return tuple([str(i) for i in [x1, x2, y1, y2]])
        elif fmt=='slice':
            # swap to fit numpy array indexing: array[y1:y2,x1:x2]
            return (slice(y1, y2), slice(x1, x2))


    def grouper(self, input_list, n=2):
        """
        Generate tuple of n consecutive items from input_list
        ex for n=2: [a,b,c,d] -> [[a,b],[b,c],[c,d]]
        """
        for i in range(len(input_list) - (n - 1)):
            yield input_list[i:i+n]
 

    def subdivide(self, subsize):
        """
        Divide the chunk into subchunk of size subsize
        Indices are in local coordinate (ie start at 0)
        """
       
        self.chunk_col = self.x1+np.array(list(range(0, self.dim[1], subsize))+[self.dim[1]])
        self.chunk_row = self.y1+np.array(list(range(0, self.dim[0], subsize))+[self.dim[0]])
   
        self.list = []
        for row0,row1 in self.parent.grouper(self.chunk_row):
            for col0,col1 in self.parent.grouper(self.chunk_col):
                # self.__class__ is used to instantiate recursively a new class
                #print(col0, col1, row0, row1) 
                self.list.append(self.__class__(col0, col1, row0, row1, self)) 

class CoordinatesConverter():
    """
    For a given format, convert a list of lat/lon pairs into :
    - list of corresponding indices in array
    - list of slice for area around the points

    Notes
    -----
    This class has to use the same public methods and output formats
    as Splitter class in order to be used independantly.
    """

    def __init__(self, fpath, sensor, sub=None):
        self.fpath = fpath
        self.sensor = sensor
        self.input = 'points'
        # return itself in a list to match Splitter format
        self.list = [self]

        ## Get the param corresponding to the resolution
        param_sensor = {'AVHRR':'4km', 'VGT':'1km', 'PROBAV':'1km'}
        param_resol = {'4km':(4200, 10800, 1), '1km':(15680, 40320, 4), '300m':('xxx', 'xxx', 12)}
        self.c3s_resol = param_sensor[self.sensor]
        self.lat_len, self.lon_len, self.box_size = param_resol[self.c3s_resol]
        
        self._load_coor_from_csv(sub)
        self._coor_to_indices_c3s()

    def _load_coor_from_csv(self, sub=None):
        """
        Load reference sites where to perform quality monitoring from a csv file.
        CSV file must have a LATITUDE and LONGITUDE columns.
        Parameters
        sub: - int: load the first sub rows
             - list of string: load sites with NAME in the list
        """
        df = pd.read_csv(self.fpath, sep=';', index_col=0)

        if sub is None:
            self.site_coor = df
        elif isinstance(sub, int):
            self.site_coor = df[:nsub]
        elif isinstance(sub, list):
            self.site_coor = df.loc[df['NAME'].isin(sub)]

    def _coor_to_indices_c3s(self):
        """
        Notes
        -----
    
        c3s data:
        - shape is (1, lat_len, lon_len)
        - lat extend from -60 to 80 deg
        - lon extend from -180 to 180 deg
        
        Here we want to extract:
        - 1x1 px for 4km
        - 4x4 px for 1km
        - 12x12 px for 300m
        """
        ## Convert lat/lon to array indices
        df = self.site_coor
        df['ilat'] = np.rint( self.lat_len * (1-(df.LATITUDE + 60)/140) ).astype(int) # data from -60 to +80 deg lat
        df['ilon'] = np.rint( self.lon_len *    (df.LONGITUDE+180)/360  ).astype(int) # data from -180 to 180 deg lon
        
        df = df.sort_values(by=['ilat', 'ilon'])
    
        ## Make a bounding box around the coordinate
        half_size = int(self.box_size/2.)
        if (self.box_size%2)==0: 
            #self.slice = [(0, lat, lon) for lat,lon in df[['ilat', 'ilon']].values]
            self.slice = [(0, slice(ilat-half_size, ilat+half_size), slice(ilon-half_size, ilon+half_size)) for ilat,ilon in df[['ilat', 'ilon']].values]
        else:
            #self.slice = [(0, slice(lat-self.box_offset, lat+self.box_offset), slice(lon-self.box_offset, lon+self.box_offset)) for lat,lon in df[['ilat', 'ilon']].values]
            self.slice = [(0, slice(ilat-half_size, ilat+half_size+1), slice(ilon-half_size, ilon+half_size+1)) for ilat,ilon in df[['ilat', 'ilon']].values]
        
        self.site_coor = df
        
        # Add parameters to fake 2D chunk
        self.dim = (1, len(self.slice)) 
        self.x1 = 0
        self.x2 = len(self.slice)
        self.y1 = 0
        self.y2 = 1

    def get_limits(self, ref, fmt):
        x1 = self.x1 
        x2 = self.x2 
        y1 = self.y1 
        y2 = self.y2 
        if fmt=='tuple':
            return (x1, x2, y1, y2)
        elif fmt=='str':
            # Use (x,y) order to print
            return tuple([str(i) for i in [x1, x2, y1, y2]])
        elif fmt=='slice':
            return self.slice

    def get_row_by_name(self, site_name):
        return self.site_coor.loc[self.site_coor['NAME'] == site_name]
        
    def get_box_around(self, site_name, size):
        """
        Return lat and lon slices to pick up a box of shape (size, size) around a site
        """
        site_data = self.get_row_by_name(site_name)
        ilat,ilon = site_data[['ilat', 'ilon']].values[0] # here values return [[xx,yy]]
        half_size = int(size/2.)
        if (size%2)==0: 
            return (slice(ilat-half_size, ilat+half_size), slice(ilon-half_size, ilon+half_size))
        else:
            return (slice(ilat-half_size, ilat+half_size+1), slice(ilon-half_size, ilon+half_size+1))

    def get_box_from_topleft(self, site_name, size):
        """
        Return lat and lon slices to pick up a box of shape (size, size) around a site
        """
        site_data = self.get_row_by_name(site_name)
        ilat,ilon = site_data[['ilat', 'ilon']].values[0] # here values return [[xx,yy]]
        return (slice(ilat, ilat+size), slice(ilon, ilon+size))


class Product():
    ## Sensor dates
    #NOAA_satellite Start_date End_date
    sensor_dates = {}
    sensor_dates['AVHRR']  = ('20-09-1981', '31-12-2005')
    sensor_dates['VGT']    = ('10-04-1998', '31-05-2014')
    sensor_dates['PROBAV'] = ('31-10-2013', '30-06-2020')
    sensor_dates['MERGED'] = ('20-09-1981', '30-06-2020')

    sensor_dates = {k: [datetime.strptime(i, "%d-%m-%Y") for i in v] for k,v in sensor_dates.items()}

    def __init__(self, product_name, start_date, end_date, chunks):
        assert isinstance(product_name, str)
        assert isinstance(start_date, datetime)
        assert isinstance(end_date, datetime)
        self.name = product_name
        self.sensor = product_name.split('_')[-1]
        self.shorten = '_'.join(product_name.split('_')[:-1])
        ## Check if date range overlaps
        if (start_date > self.sensor_dates[self.sensor][1]) or (end_date < self.sensor_dates[self.sensor][0]):
            self.start_date = None
            self.end_date = None
            self.hash = '000000'
            print("WARNING: Sensor {} is not available in this time range".format(self.sensor))
        ## if ok trim dates for each product
        else:
            self.start_date = max(start_date, self.sensor_dates[self.sensor][0])
            self.end_date = min(end_date, self.sensor_dates[self.sensor][1])
            if isinstance(chunks, str):
                self.hash = chunks
            else:
                self.hash = get_case_hash(self.name, self.start_date, self.end_date, chunks)

    def infos(self):
        pprint(vars(self))
        


def plot2Darray(v, var='var'):
    """Debug function to plot a 2D array in terminal"""
    import matplotlib.pyplot as plt

    plt.imshow(v)
    imgname = 'h52img_{0}.png'.format(var)
    plt.savefig(imgname)
    print('Saved to {}.'.format(imgname))
    os.system('/mnt/lfs/d30/vegeo/fransenr/CODES/tools/TerminalImageViewer/src/main/cpp/tiv ' + imgname)

def binned_statistic_dd(sample, values, statistic='mean',
                        bins=10, rrange=None, expand_binnumbers=False):
    """
    Compute a multidimensional binned statistic for a set of data.
    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.
    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of D arrays of length N, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:
          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'std' : compute the standard deviation within each bin. This 
            is implicitly calculated with ddof=0.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.
    bins : sequence or int, optional
        The bin specification must be in one of the following forms:
          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... = bins).
          * The number of bins for all dimensions (nx = ny = ... = bins).
    rrange : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section of
        `binned_statistic_2d`.
        .. versionadded:: 0.17.0
    Returns
    -------
    statistic : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    binnumber : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.
    See Also
    --------
    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d
    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open in each dimension.  In
    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
    last bin, however, is ``[3, 4]``, which *includes* 4.
    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    If 'True': The returned `binnumber` is a shape (D,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.
    .. versionadded:: 0.11.0
    """
    known_stats = ['mean', 'median', 'count', 'sum', 'std','min','max']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError('invalid statistic %r' % (statistic,))

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape
        #print Dlen, Ndim

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if(statistic != 'count' and Vlen != Dlen):
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    nbin = np.empty(Ndim, int)    # Number of bins in each dimension
    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if rrange is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        smin = np.zeros(Ndim)
        smax = np.zeros(Ndim)
        for i in range(Ndim):
            smin[i], smax[i] = rrange[i]

    # Make sure the bins have a finite width.
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    # Compute the bin number each sample falls into, in each dimension
    sampBin = [
        np.digitize(sample[:, i], edges[i])
        for i in range(Ndim)
    ]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in range(Ndim):
        # Find the rounding precision
        decimal = int(-np.log10(dedges[i].min())) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(np.around(sample[:, i], decimal) ==
                           np.around(edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    result = np.empty([Vdim, nbin.prod()], float)

    if statistic == 'mean':
        result.fill(np.nan)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic == 'std':
        result.fill(0)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            flatsum2 = np.bincount(binnumbers, values[vv] ** 2)
            result[vv, a] = np.sqrt(flatsum2[a] / flatcount[a] -
                                    (flatsum[a] / flatcount[a]) ** 2)
    elif statistic == 'count':
        result.fill(0)
        flatcount = np.bincount(binnumbers, None)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic == 'sum':
        result.fill(0)
        for vv in range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            result[vv, a] = flatsum
    elif statistic == 'median':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in range(Vdim):
                result[vv, i] = np.median(values[vv, binnumbers == i])
    elif statistic == 'min':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in range(Vdim):
                result[vv, i] = np.min(values[vv, binnumbers == i])
    elif statistic == 'max':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in range(Vdim):
                result[vv, i] = np.max(values[vv, binnumbers == i])
    elif callable(statistic):
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                null = statistic([])
            except Exception:
                null = np.nan
        result.fill(null)
        for i in np.unique(binnumbers):
            for vv in range(Vdim):
                result[vv, i] = statistic(values[vv, binnumbers == i])

    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if(expand_binnumbers and Ndim > 1):
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`reulst`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    #return BinnedStatisticddResult(result, edges, binnumbers)
    return (result, edges)

def get_case_hash(product, start, end, chunks):
    """ Return hash of the case from string parameters
    Args:
        prod: string name of the product
        start,end: datetime object
        lim: chunks object
    """
    kwarg = {}
    start = start.isoformat()
    end = end.isoformat()
    limits = ','.join(chunks.get_limits('global', 'str'))
    str_hash = '{0},{1},{2},{3}'.format(product, start, end, limits)
    #print(str_hash)
    hex_hash = hashlib.sha1(str_hash.encode("UTF-8")).hexdigest()

    return hex_hash[:6]



