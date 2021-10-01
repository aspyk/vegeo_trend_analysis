from datetime import *
import pandas as pd
import time
import numpy as np


freq = 5

now = datetime.now()

sec = now.hour*3600 + now.minute*60 + now.second

print(sec)
print(sec%freq, sec//freq)

sec = freq*(sec//freq + 1)

sec_val = sec % (24 * 3600)
hour_val = sec_val // 3600
sec_val %= 3600
min_val = sec_val // 60
sec_val %= 60

print(hour_val, min_val, sec_val)

tref = datetime(now.year, now.month, now.day, hour_val, min_val, sec_val)

while 1:

    time.sleep((tref-datetime.now()).total_seconds())

    print(datetime.now())
    tref += timedelta(seconds=freq)
