import pandas as pd
import numpy as np

def calculate_distance_matrix(df) -> pd.DataFrame():
  unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel())
  distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
  np.fill_diagonal(distance_matrix.values, 0)
  for _, row in df.iterrows():
    start, end, distance = row['id_start'], row['id_end'], row['distance']
    distance_matrix.at[start, end] = distance
    distance_matrix.at[end, start] = distance
  for k in unique_ids:
    for i in unique_ids:
      for j in unique_ids:
        distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])
    
  return distance_matrix



def unroll_distance_matrix(df) -> pd.DataFrame():
  unrolled_data = []

  for id_start in df.index:
    for id_end in df.columns:
      if id_start != id_end:  
        distance = df.at[id_start, id_end]
        unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
  unrolled_df = pd.DataFrame(unrolled_data)
  return unrolled_df



def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame():
  reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
  lower_bound = reference_avg_distance * 0.9
  upper_bound = reference_avg_distance * 1.1

  avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
  within_threshold = avg_distances[
    (avg_distances['distance'] >= lower_bound) & (avg_distances['distance'] <= upper_bound)
    ]
  within_threshold_sorted = within_threshold.sort_values(by='id_start').reset_index(drop=True)
  return within_threshold_sorted



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6
    return df


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    weekday_intervals = [
        (datetime.time(0, 0), datetime.time(10, 0), 0.8),
        (datetime.time(10, 0), datetime.time(18, 0), 1.2),
        (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8)
    ]
    
    weekend_intervals = [
        (datetime.time(0, 0), datetime.time(23, 59, 59), 0.7)
    ]
    
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekends = ["Saturday", "Sunday"]
    
    rows = []
    
    for _, row in df.iterrows():
        for day in weekdays + weekends:
            if day in weekdays:
                intervals = weekday_intervals
            else:
                intervals = weekend_intervals
            
            for start_time, end_time, discount in intervals:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = start_time
                new_row['end_time'] = end_time
                
                new_row['moto'] = row['moto'] * discount
                new_row['car'] = row['car'] * discount
                new_row['rv'] = row['rv'] * discount
                new_row['bus'] = row['bus'] * discount
                new_row['truck'] = row['truck'] * discount
                
                rows.append(new_row)
    
    result_df = pd.DataFrame(rows)
    
    return result_df

