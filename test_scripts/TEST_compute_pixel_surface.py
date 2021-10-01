import numpy as np
import h5py


"""
source: https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
The computed area is the surface of the ring from the equator to the latitude f.
As a validation, if f = 90deg, we have the half of the total surface of the earth. Computing the radius
back with 4*pi*r^2 we get back 6371.007 km. 

Note:
log(zp/zm) can be replaced by 2*atanh(e*sin(f)) to reduce round error.
"""

def compute_ring_area(lat):
    """
    Compute area between equator (lat=0) to lat
    """
    a = 6378.137
    b = 6356.7523142
    
    e = np.sqrt(1 - (b/a)**2)
    
    f = np.radians(lat)
    
    zm = 1 - e*np.sin(f)
    zp = 1 + e*np.sin(f)
    area = np.pi * b**2 * (np.log(zp/zm) / (2*e) + np.sin(f) / (zp*zm))
    
    return area

def validate_area():
    total_earth_area = 2*compute_ring_area(90.)
    radius = np.sqrt(total_earth_area/(4*np.pi))
    print('Earth radius [km]:', radius)

def get_pixel_area():
    """
    Suppose square pixels
    pixel_size in degrees
    """
    
    sensor = 'AVHRR'
    #sensor = 'VGT'

    if sensor=='AVHRR':
        f = '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_4KM_V2/V2.0.1/1981/c3s_ALBB-DH_19810920000000_GLOBE_AVHRR-NOAA7_V2.0.1/c3s_ALBB-DH_19810920000000_GLOBE_AVHRR-NOAA7_V2.0.1.nc'

    elif sensor=='VGT':
        f = '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_1KM_V2/V2.0.1/2000/c3s_ALBB-DH_20000610000000_GLOBE_VGT_V2.0.1/c3s_ALBB-DH_20000610000000_GLOBE_VGT_V2.0.1.nc'

    with h5py.File(f, 'r') as h:
        lat = h['lat'][:]
        dlat = np.diff(h['lat']).mean()/2.
        nlon = h['lon'].shape[0]
    
    ylat = np.linspace(lat[0]-dlat, lat[-1]+dlat, len(lat)+1)

    print(ylat)
    print(nlon)

    a = 6378.137
    b = 6356.7523142
    
    e = np.sqrt(1 - (b/a)**2)
    print(e)
    e = 0.08181919084296
    print(e)
    
    f = np.radians(np.abs(ylat))
    
    zm = 1 - e*np.sin(f)
    zp = 1 + e*np.sin(f)
    area = np.pi * b**2 * ( np.log(zp/zm)/(2*e) + np.sin(f)/(zp*zm) )
    print(area)
    area *= np.sign(ylat)
    print(area)
    area = np.abs(np.diff(area)/nlon)
    print(area)
    print(area.max())

    
    with h5py.File('output_snow/pixel_area_{}.h5'.format(sensor), 'w') as h:
        h['area'] = area
        h['data_dim'] = np.array([len(area), nlon])

if __name__=='__main__':

    validate_area()

    print(compute_ring_area(80)+compute_ring_area(60))
    
    #get_pixel_area()
