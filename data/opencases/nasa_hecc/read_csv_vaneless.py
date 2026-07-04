import numpy as np
import csv

speedlines = {}

with open('./vaneless_short_data.csv', newline='\n') as data:
    reader = csv.reader(data)
    keys = next(reader)
    curr_speedline_data = {k: [] for k in keys}

    for line in reader:
        if not line:
            mean_speed = np.mean(curr_speedline_data[keys[0]])
            speedlines[round(mean_speed)] = curr_speedline_data

            # Reset the state
            curr_speedline_rpms = []
            curr_speedline_data = {k: [] for k in keys}

            continue

        for i, k in enumerate(keys):
            curr_speedline_data[k].append(
                float(line[i]),
            )
