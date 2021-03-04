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
  - 300m in the future


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

### Outputs
TODO

### Core pipeline
TODO

### Automation
TODO

## Description of the modules

### Config file

### Reading module
#### C3S ALBEDO
Discard pixels (ie apply fill value `-32767`) in the analysis when:
- Outside valid range in AL_* ([0, 10000])
- QFLAG indicates `sea` or `continental water` (QFLAG bits 0-1)
- QFLAG indicates the algorithm failed (QFLAG bit 7)
- *_ERR > 0.2
- AGE > 30

#### C3S LAI and FAPAR
Discard pixels (ie. apply fill value `65535`) in the analysis when:
- Outside valid range in LAI / fAPAR ([0, 65534])
- Fill value in QFLAG (0)
- QFLAG indicates `obs_is_fillvalue` (QFLAG bit 0)
- QFLAG indicates `tip_untrusted` (QFLAG bit 6)
- QFLAG indicates `obs unusable` (QFLAG bit 7)


### Merging module

### Trend module

### Plotting module

## Usage

### Manual

### Automatic