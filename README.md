# Trend analysis

## Setup
```
#git clone https://github.com/aspyk/vegeo_trend_analysis
# Directly clone the dev branch
git clone --single-branch --branch feature/genericreader https://github.com/aspyk/vegeo_trend_analysis
cd vegeo_trend_analysis/
./run_setup.sh
```

## Test
```
pytest test.py -k "AVHRR_plot_all"
pytest test.py -k "S3_extract_all"
```


## Usage

```
python main.py -t0 <start_date> -t1 <end_date> -zc <xmin> <xmax> <ymin> <ymax> -p <product> -a <action>
```

With:
- `start_date` and `end_date` in ISO format `YYYY-MM-DD`
- `xmin`, `xmax`, `ymin`, `ymax` integers for coordinates of corners of a rectangular area on the MSG disk
- `product`: one or several product to process in `albedo, lai, lsf, evapo, dssf`
- `action`: which processing to apply on the product in `extract, append, trend, merge, plot`

Example:
```
python main.py -t0 2018-01-01 -t1 2018-01-31 -zc 2000 2200 300 550 -p albedo lai -a extract trend
```

You can use shortcuts to analyse specific area on the MSG disk with the `-zn` arg. Possibilities are `Euro, NAfr, SAfr, SAme`.

Example:
```
python main.py -t0 2018-01-01 -t1 2018-01-31 -zn Euro -p albedo lai -a extract trend
```

## Zone

_TODO_


## Output plot example

_TODO_
