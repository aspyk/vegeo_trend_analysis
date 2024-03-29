import numpy as np 

from bokeh.io import output_notebook, save
from bokeh.plotting import figure

from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider, HoverTool

import h5py
import pandas as pd
import os,sys
import pathlib
from timeit import default_timer as timer

class BokehPlot():

    def __init__(self):
        ## Plot land mask
        with h5py.File('c3s_land_mask.h5', 'r') as h:
            lon = h['lon'][:]
            lat = h['lat'][:]
            self.mask = h['mask'][0,:,:]
    
        self.dlon = (lon[1]-lon[0])/2.
        self.dlat = (lat[1]-lat[0])/2.
        self.extent = [lon[0]-self.dlon, lon[-1]+self.dlon, lat[-1]+self.dlat, lat[0]-self.dlat]

        self.lon = lon 
        self.lat = lat

    def rebin(self, a, shape):
        """Downsample a 2D numpy array"""
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1) 

    def plot_trends_scatter_bokeh(self):
        lon = self.lon
        lat = self.lat
        dw = lon[-1]-lon[0]
        dh = lat[0]-lat[-1]
        
        
        p = figure(plot_width=int(400.*dw/dh), plot_height=400, match_aspect=True,
                   tools="pan,wheel_zoom,box_zoom,tap,reset",
                   output_backend="webgl")

        ##--- Add land mask

        # must give a vector of image data for image parameter
        mask = self.rebin(self.mask, (int(self.mask.shape[0]/20), int(self.mask.shape[1]/20)))
        #p.image(image=[np.flipud(self.mask[::20,::20])], 
        p.image(image=[np.flipud(mask)], 
                x=lon[0], y=lat[-1], dw=dw, dh=dh,
                palette=('#FFFFFF', '#EEEEEE', '#DDDDDD', '#CCCCCC', '#BBBBBB', '#AAAAAA', '#999999', '#888888'), level="image")
        p.grid.grid_line_width = 0.5
    
        ##--- Read csv, filter and convert to ColumnDataSource

        df = pd.read_csv('output_plot/C3S_AL_BBDH_19810920_20200630/C3S_AL_BBDH_19810920_20200630.csv', sep=';', index_col=0)
        df['id2'] = np.arange(df.shape[0])
        print("df.shape=", df.shape)
        df = df.dropna(subset=['AL_DH_BB_sn'])
        print(df)
        #df = df.drop(df[(df['AL_DH_BB_sn'] ) & (df.score > 20)].index)

        dfs = df[['LONGITUDE', 'LATITUDE', 'AL_DH_BB_sn', 'id2']]
        ## Create two sources: one to keep all the points and one that will be populated according to slider
        s_ori = ColumnDataSource(dfs)
        source = ColumnDataSource(data=dict(LONGITUDE=[], LATITUDE=[], AL_DH_BB_sn=[], id2=[]))

        ##--- Modify seismic colormap

        from bokeh.models import LinearColorMapper, ColorBar
        import matplotlib.cm as mcm
        import matplotlib.colors as mcol

        fcmap = mcm.get_cmap('seismic')
        cmap_mod = [fcmap(i) for i in np.linspace(0,1,15)]
        cmap_mod[7] = mcm.get_cmap('RdYlGn')(0.5) # replace white in the middle by the yellow of RdYlGn
        scmap = mcol.LinearSegmentedColormap.from_list("", cmap_mod) # recreate a colormap
        ## Extract 256 colors from the new colormap and convert them to hex
        cmap_mod = [scmap(i) for i in np.linspace(0,1,256)]
        cmap_mod = ["#%02x%02x%02x" % (int(255*r), int(255*g), int(255*b)) for r, g, b, _ in cmap_mod]
        ## Make a colormapper based on the previous 256 colors (needed because it does not make linear interpolation between colors)
        sn_max = np.abs(np.nanmax(df['AL_DH_BB_sn']))
        color_mapper = LinearColorMapper(palette=cmap_mod, low=-sn_max, high=sn_max)

        ##--- Add scatter

        scatter_renderer = p.scatter(x='LONGITUDE', y='LATITUDE', size=12,
                                     color={'field': 'AL_DH_BB_sn', 'transform': color_mapper},
                                     source=source)

        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
        p.add_layout(color_bar, 'right')

        ## Add hover tool that only act on scatter and not on the background land mask
        p.add_tools(
            HoverTool(
                #tooltips=[("A", "@A"), ("B", "@B"), ("C", "@C")], mode = "vline"
                renderers=[scatter_renderer]
            )
        )

        ##--- Add slider

        slider = Slider(start=0.0, end=sn_max*1000., value=0.0, step=sn_max*1000./20., title="Threshold [10e-3]")
        
        update_scatter = CustomJS(args=dict(source=source, slider=slider, s_ori=s_ori), code="""
            var data_ori =  s_ori.data;
            var f    =  slider.value;
            var lon  =  data_ori['LONGITUDE'];
            var lat  =  data_ori['LATITUDE'];
            var c    =  data_ori['AL_DH_BB_sn'];
            var id    =  data_ori['id2'];
            const data = {'LONGITUDE':[], 'LATITUDE':[], 'AL_DH_BB_sn':[], 'id2':[]};
            //const indices = cb_data.index.indices
            //for (var i = 0; i < indices.length; i++) {
            //    data['x0'].push(circle.data.x[start])
            //    data['y0'].push(circle.data.y[start])
            //    data['x1'].push(circle.data.x[end])
            //    data['y1'].push(circle.data.y[end])
            //}
            for (var i = 0; i < lon.length; i++) {
                if (Math.abs(c[i])>=0.001*f) {
                    data['LONGITUDE'].push(lon[i]);
                    data['LATITUDE'].push(lat[i]);
                    data['AL_DH_BB_sn'].push(c[i]);
                    data['id2'].push(id[i]);
                }
            }
            source.data = data
        """)
        slider.js_on_change('value', update_scatter)
        
        ##--- Add time series of selected point      

        hf = h5py.File('output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5', 'r')
        ts = hf['vars/AL_DH_BB'][:,0,:].T
        dates = hf['meta/global_id'][:]

        print(ts.shape)

        p2 = figure(plot_width=int(400.*dw/dh), plot_height=400,
                   tools="pan,wheel_zoom,box_zoom,hover,reset",
                   output_backend="webgl")
        al = ts[500]
        p2.line(x=dates, y=al)

        ##--- Save html file

        save(column(slider, p, p2))
        #save(p)

    def test_image(self):
        N = 500
        x = np.linspace(0, 10, N)
        y = np.linspace(0, 10, N)
        xx, yy = np.meshgrid(x, y)
        d = np.sin(xx)*np.cos(yy)
        
        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        p.x_range.range_padding = p.y_range.range_padding = 0
        
        # must give a vector of image data for image parameter
        p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
        p.grid.grid_line_width = 0.5

        save(p)

if __name__=='__main__':

    b = BokehPlot()
    #b.test_image()
    
    b.plot_trends_scatter_bokeh()


    
        










   
