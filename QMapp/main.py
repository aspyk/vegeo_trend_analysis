import numpy as np 

from bokeh.io import save, curdoc
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider, HoverTool, TapTool, BoxAnnotation

from datetime import datetime
import h5py
import pandas as pd
import os,sys
import pathlib
from timeit import default_timer as timer


class BokehPlot():

    def __init__(self):
        self.app_dir = pathlib.Path(__file__).parent

        ## Plot land mask
        with h5py.File(self.app_dir/'data/c3s_land_mask.h5', 'r') as h:
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

        df = pd.read_csv(self.app_dir/'data/output_plot/C3S_AL_BBDH_19810920_20200630/C3S_AL_BBDH_19810920_20200630.csv', sep=';', index_col=0)
        df['id2'] = np.arange(df.shape[0])
        print("df.shape=", df.shape)
        df = df.dropna(subset=['AL_DH_BB_sn'])
        print(df)
        #df = df.drop(df[(df['AL_DH_BB_sn'] ) & (df.score > 20)].index)

        dfs = df[['LONGITUDE', 'LATITUDE', 'AL_DH_BB_sn', 'id2']]
        ## Create two sources: one to keep all the points and one that will be populated according to slider
        s_ori = ColumnDataSource(dfs)
        #source = ColumnDataSource(data=dict(LONGITUDE=[], LATITUDE=[], AL_DH_BB_sn=[], id2=[]))
        source = ColumnDataSource(dfs)

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
                renderers=[scatter_renderer],
                mode='mouse'
            )
        )

        ##--- Add slider

        slider = Slider(start=0.0, end=sn_max*1000., value=0.0, step=sn_max*1000./20., title="Threshold [10e-3]")
        
        ## Python callback
        def update_scatter(attr, old, new):
            # new = new slider value 
            source.data = ColumnDataSource.from_df(dfs.loc[np.abs(dfs['AL_DH_BB_sn']) >= 0.001*new ])

        slider.on_change('value', update_scatter)
        
        ##--- Add time series of selected point      
        
        hf = h5py.File(self.app_dir/'data/output_extract/c3s_al_bbdh_MERGED/timeseries_198125_202017.h5', 'r')
        ts = hf['vars/AL_DH_BB'][:,0,:].T
        dates = hf['meta/ts_dates'][:].view('datetime64[s]').tolist()
        #dates = hf['meta/global_id'][:]
        #dates = 1970+dates//36 + (10*(1+dates%36))/365.

        pw = int(400.*dw/dh)
        ph = 200
        p2 = figure(plot_width=pw, plot_height=ph,
                   tools="pan,wheel_zoom,box_zoom,reset",
                   output_backend="webgl", x_axis_type="datetime")



        p2.add_tools(
        HoverTool(tooltips=[
            ("Date", "@dates{%Y-%m-%d}"),  # must specify desired format here
            ("Value", "@var")
        ], formatters={"@dates":"datetime"}, mode='vline')
        )

        ## Create source and plot it

        #ts_source = ColumnDataSource(data=dict(dates=[], var=[]))
        d_init = [d for d in dates if d.year != 1970]
        ts_source = ColumnDataSource(data=dict(dates=d_init, var=np.zeros_like(d_init)))
        p2.line(x='dates', y='var', source=ts_source)
        

        ## Add satellite periods
        # Sensor dates
        sensor_dates = []
        sensor_dates.append(['NOAA7',  ('20-09-1981', '31-12-1984')])
        sensor_dates.append(['NOAA9',  ('20-03-1985', '10-11-1988')])
        sensor_dates.append(['NOAA11', ('30-11-1988', '20-09-1994')])
        sensor_dates.append(['NOAA14', ('10-02-1995', '10-03-2001')])
        sensor_dates.append(['NOAA16', ('20-03-2001', '10-09-2002')])
        sensor_dates.append(['NOAA17', ('20-09-2002', '31-12-2005')])
        sensor_dates.append(['VGT1',   ('10-04-1998', '31-01-2003')])
        sensor_dates.append(['VGT2',   ('31-01-2003', '31-05-2014')])
        sensor_dates.append(['PROBAV', ('31-10-2013', '30-06-2020')])
        sensor_dates = [[v[0], [datetime.strptime(i, "%d-%m-%Y") for i in v[1]]] for v in sensor_dates]

        import itertools
        from bokeh.palettes import Category10 as palette
        colors = itertools.cycle(palette[10])
        top_ba = []
        bottom_ba = []
        for v,color in zip(sensor_dates, colors):
            if 'VGT' not in v[0]:
                top_ba.append(BoxAnnotation(top=ph, top_units='screen', bottom=int(ph/2), bottom_units='screen',
                                   left=v[1][0], right=v[1][1], fill_alpha=0.2, fill_color=color))
            else:
                bottom_ba.append(BoxAnnotation(top=int(ph/2), top_units='screen', bottom=0, bottom_units='screen',
                                   left=v[1][0], right=v[1][1], fill_alpha=0.2, fill_color=color))
        if 1:
            for ba in top_ba:
                p2.add_layout(ba)
            for ba in bottom_ba:
                p2.add_layout(ba)

            ## Plot name
            #if tmin < 0.5*(v[1][1]+v[1][0]) < tmax:
            #    t = plt.text(0.5*(v[1][1]+v[1][0]), 1.*dmax, v[0], horizontalalignment='center', fontsize=8)


        def update_ts(attr, old, new):
            """
            attr: 'indices'
            old (list): the previous selected indices
            new (list): the new selected indices
            """
            if 0:
                print(p2.width, p2.height)
                print(p2.frame_width, p2.frame_height)
                print(p2.inner_width, p2.inner_height)
                print(p2.x_range.start, p2.x_range.end, p2.x_scale)
                print(p2.y_range.start, p2.y_range.end, p2.y_scale)

            if len(new)>0:
                ## Update line
                site_id = source.data['id2'][new[0]]
                ts_source.data = dict(dates=dates, var=ts[site_id])
                
                ## Update BoxAnnotation
                ph = p2.inner_height
                for ba in top_ba:
                    ba.top = ph
                    ba.bottom = int(ph/2)
                for ba in bottom_ba:
                    ba.top = int(ph/2)
                    ba.bottom = 0

        source.selected.on_change('indices', update_ts)

        ##--- Save html file

        #save(column(slider, p, p2))
        #save(p)

        ##--- Serve the file
        
        curdoc().add_root(column(slider, p, p2))
        curdoc().title = "Quality monitoring"

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

print(__name__)

# if __name__=='__main__':
if __name__.startswith('bokeh_app'):

    b = BokehPlot()
    #b.test_image()
    
    b.plot_trends_scatter_bokeh()


    
        










   
