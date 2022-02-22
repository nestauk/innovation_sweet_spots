## Producing time series data

To produce time series run the scripts in this directory from the terminal.

`timeseries_cb.py` needs an argument relating to what time period (either month, quarter or year) to group the data by.
For example, to produce the crunchbase time series with the data grouped montly, run this from terminal:

```
 python timeseries_cb.py month
```

`timeseries_gtr.py` needs an argument relating to what time period (either month, quarter or year) to group the data by. It also has an optional flag (`--split`) whether to split the research funding more evenly across the duration of research projects.
For example, to produce the GtR data grouped yearly and have the research funding split, run this from terminal:

```
 python timeseries_gtr.py year --split
```
