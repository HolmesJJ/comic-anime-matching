import os.path

import pandas as pd

from datetime import datetime
from datetime import timedelta


OUTPUT_DIR = 'output'


def parse_timestamp(timestamp, frame_rate=24):
    try:
        time_part, frames = timestamp[:-3], int(timestamp[-2:])
        parsed_time = datetime.strptime(time_part, "%H:%M:%S") + timedelta(milliseconds=(frames * 1000 / frame_rate))
        return parsed_time
    except Exception as e:
        print('Invalid timestamp', e)
        return None


def check_timestamps(data_df, frame_rate=24, tolerance=1e-5):
    data_df['Parsed Start Timestamp'] = data_df['Start Timestamp'].apply(
        lambda x: parse_timestamp(x, frame_rate) if pd.notnull(x) else None)
    data_df['Parsed End Timestamp'] = data_df['End Timestamp'].apply(
        lambda x: parse_timestamp(x, frame_rate) if pd.notnull(x) else None)
    violations = []
    for i in range(len(data_df) - 1):
        current_end = data_df.at[i, 'Parsed End Timestamp']
        next_start = data_df.at[i + 1, 'Parsed Start Timestamp']
        if current_end and next_start:
            expected_next_start = current_end + timedelta(milliseconds=(1000 / frame_rate))
            if abs((next_start - expected_next_start).total_seconds()) > tolerance:
                if not pd.isnull(data_df.at[i + 1, 'Comic Block ID']):
                    violations.append((i, i + 1))
    for curr_idx, next_idx in violations:
        print(f"Violation between rows {curr_idx} and {next_idx}:")
        print(data_df.iloc[[curr_idx, next_idx]][['Comic Block ID', 'Start Timestamp', 'End Timestamp']])


def run():
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, '001.csv'))
    check_timestamps(data_df)


if __name__ == '__main__':
    print(parse_timestamp('00:03:37:18'))
    run()
