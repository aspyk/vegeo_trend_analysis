---
title: C3S Quality Monitoring Tool
---
<link rel="stylesheet" href="./css/custom.css">
<!--
<link rel="stylesheet" href="./css/modest.css">
-->

# C3S Quality Monitoring Tool

## Objective of the tool
The objective of the tool is to compute albedo, LAI and FAPAR trends based on Mann Kendall test on copernicus v2 data. This tool will used about every 6 months in order to update the trend.


## Overview of the core pipeline
The structure of the tool is composed of a main core pipeline running on a single product, this pipeline looping then on a list of desired products products.

### Inputs
#### C3S data
C3S data are available since 1981 and continue today. Three new global maps are given each month (every ~10 days), giving a total of 36 files for each full year. 
Until now, these files may have several resolutions coming from different sensors :
  - 4km resolution giving a 4200x10800 map (AVHRR sensor)
  - 1km resolution giving a 15680x40320 map (VGT and PROBAV sensors)
  - 300m in the future with SENTINEL3


<figure style="text-align:center">
  <img src="./img/sensor_timeline.png" alt="sensor_timeline.png"/>
  <figcaption>C3S sensor timeline.</figcaption>
</figure>

#### Reference sites
Reference sites are ~700 points located all over the world. This input is given as a csv file with the following structure:
```
"#";LATITUDE;LONGITUDE;NAME
1;-10.76;-62.3583;ABRACOS_HILL
2;51.75;59.75;ADAMOWKA
[...]
```

<figure style="text-align:center">
  <img src="./img/ref_sites.png" alt="ref_sites.png"/>
  <figcaption>All the reference sites on the C3S land mask.</figcaption>
</figure>

#### YAML config file
Path and options are given in a config file using the YAML format.

### Outputs
Outputs of the code are of three types:
- cache files

These files are written between each modules in order to save previous work for being unnecessarily reprocessed, especially for the extract part that can take some time. Cache files have HDF5 format and can be easily read outside of the code if required. Detailed description is given in the module section.

- PNG images

PNG images are written at the end of the last module to plot the result of the trend computation. The format use a scatter plot on a global map with point colored by the value of the slope computed by the Theil-Sen estimator.

<figure style="text-align:center">
  <img src="./img/output_image_example.png" alt="output_image_example.png"/>
  <figcaption>Output example where trends in [unit/year] of AL_DH_VI channel over about 20 years is displayed.</figcaption>
</figure>

- CSV files

CSV files are written at the same time as the PNG files and simply export numerical values use to plot the scatter plot as plain text to be used for further analysis if necessary. The format is the same as for the CSV input file containing the LANDVAL sites, new columns are just added to give the statistical parameters computed previously.

### Core pipeline
The core pipeline is made up of 4 modules communicating together only using cache files. These modules are the reading module, the merging module, the trend module and the plotting module. The following section is dedicated to their detailed description. 

A flowchart summarizing the whole structure is given below:

<figure style="text-align:center">
  <img src="./img/input_output_overview.png" alt="input_output_overview.png"/>
  <figcaption>Flowchart of all the modules with their inputs and outputs.</figcaption>
</figure>

## Description of the modules

TODO

### Config file
Config file use the [YAML format](https://yaml.org/). The file is divided into several parts. The first one give parameters about C3S data files, that is mainly their path (under the `root` keyword) and the variables to be analyzed (under the `vars` keyword). Others parameters (`source`, `mode`, `freq`) should not be modified in the scope of the actual C3S global datasets available. Note the use of anchors with `<<: &foo` and `<<: *foo` to avoid repeating the same parameters for several datasets. 

Below is an extract of this part for the AL_BB_DH datasets grouping the 4 sensors:
```yaml
 # AL_BB_DH
 # --------

c3s_al_bbdh_AVHRR:
    root: '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_4KM_V2/V2.0.1'
    <<: &param
        source: 'c3s'
        mode: 'walk'
        freq: '10D'
    <<: &varbbdh
        var:
            - 'AL_DH_BB'
            - 'AL_DH_NI'
            - 'AL_DH_VI'

c3s_al_bbdh_VGT:
    root: '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_1KM_V2/V2.0.1'
    <<: *param
    <<: *varbbdh

c3s_al_bbdh_PROBAV:
    root: '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_1KM_V2/V2.0.1'
    <<: *param
    <<: *varbbdh

c3s_al_bbdh_SENTINEL3:
    root: '/data/c3s_pdf_live/MTDA/C3S_ALBB_DH_Global_300M_V3/V3.0.1'
    <<: *param
    <<: *varbbdh

```

The second part at the end of the file looks like this:
```yaml
 # Dir to store cache and output
output_path:
    extract: './output_extract'
    merge: './output_merge'
    trend: './output_trend'
    plot: './output_plot'
    merged_filename: 'merged_trends.nc'

 # Input file with reference site coordinates
 # (read only if input type = latloncsv)
ref_site_coor:
    AVHRR: './LANDVAL_v1.1.csv'
    VGT: './LANDVAL_v1.1_avhrr_topleft.csv'
    PROBAV: './LANDVAL_v1.1_avhrr_topleft.csv'
    SENTINEL3: './LANDVAL_v1.1_avhrr_topleft.csv'

```
`output_path` section group the folders where cache files and output images and CSV should be written. Each module should always writes its outputs in separate folders to avoid unexpected conflicts.

### Reading module
One-dimensional time series are required for the Mann-Kendall test but inputs are using several formats and resolution (GPS coordinates for LANDAVAL sites and 4KM, 1KM and 300M resolutions for C3S datasets). Then to be able to use the whole time series for statistics computation a first aggregating pre-processing step is required to get uniform data. For that the reading module can be divided in two part: first the extraction part, then the aggregation part.

#### Extraction
One pixel from the lower resolution (~4km) is taken as a surface reference. Then, the following extraction is done for each available resolution:
-	4KM series: simple extraction of the pixel containing the reference site (or 1x1 matrix).
-	1KM series: 4x4 pixel matrix extraction, centered on the 4KM pixel extracted above (and not directly on the reference site to avoid possible large offset)
-	300M series: 12x12 pixel matrix extraction, centered as above on the 4KM pixel.

Note that to simplify the reading of the matrices, a second CSV input file has been automatically created containing the GPS coordinates of the top-left corner of the AVHRR pixel instead of the LANDVAL site initial coordinates.

TODO: So far this task is performed by the tool `TOOL_compare_grid.py`. The function writing the modified input CSV file should be integrated directly in the main code.   

#### Aggregation
Several tests are then applied on all the extracted pixels to consider them as a valid input for the Mann Kendall test. For the matrix cases, these tests are first applied on each pixel of the 4x4 and 12x12 matrices, and if there is more than 75% of valid pixels in them the final unique value is computed averaging the values of all the valid pixels.
The tests used to differentiate the pixels are the following:

Since the Mann Kendall test requires to keep the real time range between the values, resulting value of the test will be either the valid value or a NaN. Note that when two sensors overlap in time, the newer overwrites the older.
Therefore, the output of the pre-processing step is a one-dimensional time series for each LANDVAL sites made of 36 time slots per year being filled by either NaN or actual (4KM) or averaged value (1KM or 300M).

##### C3S ALBEDO
Discard pixels (ie apply fill value `-32767`) in the analysis when:
- Outside valid range in AL_* ([0, 10000])
- QFLAG indicates `sea` or `continental water` (QFLAG bits 0-1)
- QFLAG indicates the algorithm failed (QFLAG bit 7)
- *_ERR > 0.2
- AGE > 30

##### C3S LAI and FAPAR
Discard pixels (ie. apply fill value `65535`) in the analysis when:
- Outside valid range in LAI / fAPAR ([0, 65534])
- Fill value in QFLAG (0)
- QFLAG indicates `obs_is_fillvalue` (QFLAG bit 0)
- QFLAG indicates `tip_untrusted` (QFLAG bit 6)
- QFLAG indicates `obs unusable` (QFLAG bit 7)

#### Implementation

Since the Mann-Kendall analysis requires to keep the exact time range between all points, the idea here is to create an initial array full of NaN with the shape `(number_of_time_slot, number_of_sites)`, and then for each time slot, if a file is available, extract all the matrices from the file to put them in a `(number_of_sites, matrix_dim[0], matrix_dim[1])` array. Then, aggregation is applied on the latter array to get a 1D array of shape `number_of_sites` that will fill the initial array.  


In the `generic.py` file, a helper class called `CoordinatesConverter` is going to read the list of input GPS coordinates and convert it into a list of ready-to-use `slices` objects adapted to each resolution. For example the following coordinates:
```
673;71.0938;134.978;Republica_Saja_8
```
are converted to the following slice object for the VGT resolution:
```
s = (0, slice(995, 999, None), slice(35274, 35278, None))
```
This slice can then be directly used for array slicing like this:
```
hdf5_file['AL_DH_BB'][s]
```
To get directly the 4x4 matrix:
```
[[6688 6695 6697 6697]
 [6694 6669 6665 6642]
 [6648 6650 6636 6613]
 [6689 6726 6742 6724]]
```

Then, all of these matrices are aggregated in the `_get_c3s_albedo_points` method of the `TimeSeriesExtractor` class in the `time_series_reader.py` file.

The shape of the output of this extraction will be as said above a `(number_of_time_slot, number_of_sites)` array.

#### Output format
Note that this code was intended to work not only on coordinates list but also on 2D areas. Therefore, to allow the use of the same processing routines for both cases, arrays used for the present points extraction will always have 

### Merging module

### Trend module

### Plotting module

### Summary of the code structure

<figure style="text-align:center">
  <img src="./img/pipeline_overview.png" alt="pipeline_overview.png"/>
  <figcaption>Summary of modules actions.</figcaption>
</figure>

<figure style="text-align:center">
  <img src="./img/call_graph.png" alt="call_graph.png"/>
  <figcaption>Call graph of the code.</figcaption>
</figure>

## Usage

### Manual
A manual run is a run where specific time range, sensor and product are given by the user in a command line.
```
python main.py -t0 <start_date> -t1 <end_date> -i <type_of_input> -p <product_tag> -a <action> --config <config_file_path> [-d] [--debug 1]
```
Example:
```
python main.py -t0 1981-01-01 -t1 2020-12-31 -i latloncsv:config -p c3s_al_bbdh_AVHRR c3s_al_bbdh_VGT c3s_al_bbdh_PROBAV -a extract merge trend plot -d --config config_vito.yml
```

 - -t0 START_DATE : start date in iso format `YYYY-MM-DD`.
 - -t1 END_DATE : end date in iso format `YYYY-MM-DD`.
 - -p PRODUCT_TAG : whitespace separated list of tag(s) using the `c3s_<product>_<sensor>` format. `product` can be in `{al_bbdh,al_bbbh,al_spdh,al_spbh,lai,fapar}` and `sensor` in `{AVHRR,VGT,PROBAV,SENTINEL3}`.
- -i INPUT : input type and parameter(s), use the format `<type>:<param1>,<param2>...`. The appropriate input for quality monitoring on LANDVAL sites is `latloncsv:<path_to_csv_file>` but the shortcut option `latloncsv:config` allow to read the input csv file path from the YAML config file.
- -a ACTION : whitespace separated list of possible actions in `{extract,merge,trend,plot}`
- -c CONFIG : path to the YAML config file.
- [-d] DELETE_CACHE : option to force overwriting of the cache files and reprocess data.
- [-g ] DEBUG : debug option, read only a small subset of all the LANDVAL sites. User have to modify this list in `main.py` file.



### Automatic
The automatic run can be launch with a single short command. The run will loop over:
- the whole available time range, that is from 1981 to the date of the run.
- AVHRR, VGT, PROBAV and SENTINEL3 sensors.
- albedo, LAI and fAPAR products.

The command to run is the following:
```
python main_loop.py
```
After a (long) while, cache files and outputs will be available in their respective folder.