from TOOL_snht import *


### Uncomment to generate a Monte Carlo cache file (can take some times...)
#create_mc_cache(nmin=10, nmax=1600, sim=20000)

### To perform a snht test over a QM tool cache file:
qm_cache_file = '/data/c3s_vol6/TEST_CNRM/remymf_test/vegeo_trend_analysis/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5'
var = 'AL_DH_BB'
VITO_recursive_snht(qm_cache_file, var)
