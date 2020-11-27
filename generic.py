import numpy as np
import hashlib


class Splitter():
    """
    Class to handle the partitioning of the selected region into chunks.

    Args:
        xlim1,xlim2,ylim1,ylim2: global region coordinates on MSG disk.
        nmaster: the size of the chunks, 500 by default.
    """
    
    class Chunk():
        def __init__(self, parent, row0, row1, col0, col1):
            """
            global_lim/slice are in the MSG disk reference
            local_lim/slice are in the extracted region reference
            """
            self.parent = parent
            self.r0 = row0
            self.r1 = row1
            self.c0 = col0
            self.c1 = col1

            # (xdim, ydim)
            self.dim = (self.r1-self.r0, self.c1-self.c0)

            self.global_lim = (self.r0, self.r1, self.c0, self.c1)
            # swap to fit numpy array indexing: array[y1:y2,x1:x2]
            self.global_slice = (slice(*self.global_lim[2:]), slice(*self.global_lim[:2]))
    
            self.local_lim = (self.r0-self.parent.x1, self.r1-self.parent.x1, self.c0-self.parent.y1, self.c1-self.parent.y1)
            # swap to fit numpy array indexing: array[y1:y2,x1:x2]
            self.local_slice = (slice(*self.local_lim[2:]), slice(*self.local_lim[:2]))

        def get_lim(ref, fmt):
            """
            ref: 'loc' or 'glob'
            fmt: 'tuple' or 'slice'
            Not sure if needed yet..
            """
            pass


    def __init__(self, xlim1, xlim2, ylim1, ylim2, nmaster=500):
        self.x1 = xlim1
        self.x2 = xlim2
        self.y1 = ylim1
        self.y2 = ylim2
        self.nmaster = nmaster

        ## the master_chunk or nmaster slices the region and can be a number, 100, 200, 500 etc., currently tested for 500 X 500 '''
        self.chunks_row = list(range(xlim1, xlim2, nmaster))+[xlim2]
        self.chunks_col = list(range(ylim1, ylim2, nmaster))+[ylim2]
   
        self.list = []
        for col0,col1 in self.grouper(self.chunks_col):
            for row0,row1 in self.grouper(self.chunks_row):
                self.list.append(self.Chunk(self, row0,row1,col0,col1))
    
    def global_limits(self, fmt=None):
        """
        Return the global limits with the specified format

        fmt can be:
        - None    : return tuple
        - 'str'   : return a string
        - 'slice' : return slice object
        """

        if fmt is None:
            return (self.x1, self.x2, self.y1, self.y2)
        elif fmt=='str':
            return ','.join([str(i) for i in [self.x1, self.x2, self.y1, self.y2]])
        elif fmt=='slice':
            # swap to fit numpy array indexing: array[y1:y2,x1:x2]
            return (slice(self.y1, self.y2), slice(self.x1, self.x2))
        

    def grouper(self, input_list, n=2):
        """
        Generate tuple of n consecutive items from input_list
        ex for n=2: [a,b,c,d] -> [[a,b],[b,c],[c,d]]
        """
        for i in range(len(input_list) - (n - 1)):
            yield input_list[i:i+n]


def plot2Darray(v, var='var'):
    """Debug function to plot a 2D array in terminal"""
    import matplotlib.pyplot as plt

    plt.imshow(v)
    imgname = 'h52img_{0}.png'.format(var)
    plt.savefig(imgname)
    print(f'Saved to {imgname}.')
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

def get_case_hash(product, start, end, limits):
    """ Return hash of the case from string parameters
    Args:
        prod: string name of the product
        start,end: date in isoformat YYYY-MM-DD (use .isoformat() on datetime object)
        lim: comma separated region limits
    """
    str_hash = '{0},{1},{2},{3}'.format(product, start, end, limits)
    #print(str_hash)
    hex_hash = hashlib.sha1(str_hash.encode("UTF-8")).hexdigest()

    return hex_hash[:6]



