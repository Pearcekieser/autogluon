import re
import calendar
import itertools
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
from datetime import date, datetime
from pandas import DataFrame, Series 


def make_date(series: Series):
    """Make sure `series` is of the right date type."""
    field_dtype = series.dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        series = pd.to_datetime(series, infer_datetime_format=True)
    return series

def add_datepart(df: DataFrame, field_name: str, drop: bool = True, time: bool = False, add_cyclic: bool = True) -> DataFrame:
    """Helper function that adds columns relevant to a date in the column `field_name` of `df`."""
    df[field_name] = make_date(df[field_name])
    df.join(expand_date_sequential(df[field_name], drop=True, time=time))
    if add_cyclic:
        df.join(add_cyclic_datepart(df[field_name], drop=True, time=time))

def expand_date_sequential(series: Series, field_name: str = None, prefix: str = None, drop: bool = True, time: bool = False) -> DataFrame:
    """Helper function that adds columns relevant to a date series."""
    if field_name == None: field_name = series.name
    series = make_date(series)
    df = pd.DataFrame({field_name: series})
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'
    ]
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[prefix + n] = getattr(series.dt, n.lower())
    df[prefix + 'Elapsed'] = series.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df

cyclic_fns = {
    'sin': np.sin,
    'cos': np.cos
}
cyclic_value_getters = {
    'weekday': lambda day: day.weekday(),
    'day_month': lambda day: day.day-1,
    'month_year': lambda day: day.month-1,
    'day_year': lambda day: day.timetuple().tm_yday-1,
    'sec_in_year': lambda day: (day - datetime(day.year, 1, 1)).total_seconds(),
    'hour': lambda day: day.timetuple().tm_hour,
    'clock_hour': lambda day: day.timetuple().tm_hour%12,
    'minute': lambda day: day.timetuple().tm_min,
    'second': lambda day: day.timetuple().tm_sec
}
cyclic_max_value_getters = {
    'weekday': lambda day: 7,
    'day_month': lambda day: calendar.monthrange(day.year, day.month)[1],
    'month_year': lambda day: 12,
    'day_year': lambda day: 366 if calendar.isleap(day.year) else 365,
    'sec_in_year': lambda day: 
        (datetime(day.year+1, 1, 1) - datetime(day.year, 1, 1)).total_seconds(),
    'hour': lambda day: 24,
    'clock_hour': lambda day: 12,
    'minute': lambda day: 60,
    'second': lambda day: 60
}

def add_cyclic_datepart(series:Series, field_name:str=None, drop:bool=False, time:bool=False, add_linear:bool=False):
    if field_name == None: field_name = series.name
    fns = ['cos','sin']
    new_cols = ['weekday', 'day_month', 'month_year', 'day_year']
    time_cols = ['hour', 'clock_hour', 'minute', 'second']
    if time: new_cols += time_cols
    if add_linear: new_cols.append('sec_in_year')

    series = make_date(series)
    
    df = DataFrame({field_name: series})
    for col, fn in itertools.product(new_cols, fns):
        col_name = f'{field_name}_{col}_{fn}'
        df[col_name] = get_cyclic(series, col, fn)
    
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

def get_cyclic(series:Series, new_cols, fn='sin') -> Series:
    fn = cyclic_fns[fn]
    get_value = cyclic_value_getters[new_cols]
    get_max_value = cyclic_max_value_getters[new_cols]
    return series.apply(_cyclize, args=(get_value, get_max_value, fn))

def _cyclize(date, get_value, get_max_value, fn):
    max_value = get_max_value(date)
    data = get_value(date)
    return fn(2 * np.pi * data/max_value)







def cyclic_dt_feat_names(time:bool=True, add_linear:bool=False)->List[str]:
    "Return feature names of date/time cycles as produced by `cyclic_dt_features`."
    fs = ['cos','sin']
    attr = [f'{r}_{f}' for r in 'weekday day_month month_year day_year'.split() for f in fs]
    if time: attr += [f'{r}_{f}' for r in 'hour clock min sec'.split() for f in fs]
    if add_linear: attr.append('year_lin')
    return attr

def cyclic_dt_features(d:Union[date,datetime], time:bool=False, add_linear:bool=False)->List[float]:
    "Calculate the cos and sin of date/time cycles."
    tt,fs = d.timetuple(), [np.cos, np.sin]
    day_year,days_month = tt.tm_yday, calendar.monthrange(d.year, d.month)[1]
    days_year = 366 if calendar.isleap(d.year) else 365
    rs = d.weekday()/7, (d.day-1)/days_month, (d.month-1)/12, (day_year-1)/days_year
    feats = [f(r * 2 * np.pi) for r in rs for f in fs]
    if time and isinstance(d, datetime) and type(d) != date:
        rs = tt.tm_hour/24, tt.tm_hour%12/12, tt.tm_min/60, tt.tm_sec/60
        feats += [f(r * 2 * np.pi) for r in rs for f in fs]
    if add_linear:
        if type(d) == date: feats.append(d.year + rs[-1])
        else:
            secs_in_year = (datetime(d.year+1, 1, 1) - datetime(d.year, 1, 1)).total_seconds()
            feats.append(d.year + ((d - datetime(d.year, 1, 1)).total_seconds() / secs_in_year))
    return feats

def fastai_add_cyclic_datepart(series:Series, prefix:str=None, drop:bool=True, time:bool=False, add_linear:bool=False):
    "Helper function that adds trigonometric date/time features to a date series."
    series = make_date(series)
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', series.name))
    series = series.apply(partial(cyclic_dt_features, time=time, add_linear=add_linear))
    columns = [f'{prefix}_{c}' for c in cyclic_dt_feat_names(time, add_linear)]
    # add drop
    return pd.DataFrame([item for item in series], columns=columns, index=series.index)

def ifnone(a, b):
    """`a` if `a` is not None, otherwise `b`."""
    return b if a is None else a
