from typing import Dict, List, Any
import re
import pandas as pd
import polyline
import math
from datetime import datetime, timedelta

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
  for i in range(0, len(lst), n):
    start = i
    end = min(i + n - 1, len(lst) - 1)
    while start < end:
      lst[start], lst[end] = lst[end], lst[start]
      start += 1
      end -= 1
      
  return lst

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    dict1 = {}
    for i in lst:
      if len(i) not in dict1:
        dict1[len(i)]=[i]
      else:
        dict1[len(i)].append(i)
    dict1 = dict(sorted(dict1.items()))
    return dict1

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
  def _flatten(current_dict: Dict[str, Any], parent_key: str) -> Dict[str, Any]:
    new_dict = {}
    for k, v in current_dict.items():
      new_k = f"{parent_key}{sep}{k}" if parent_key else k
      if isinstance(v, dict):
        new_dict.update(_flatten(v, new_k))
      elif isinstance(v, list):
        for i, item in enumerate(v):
          list_key = f"{new_k}[{i}]"
          if isinstance(item, dict):
            new_dict.update(_flatten(item, list_key))
          else:
            new_dict[list_key] = item
      else:
        new_dict[new_k] = v
    return new_dict

  return _flatten(nested_dict, '')


def unique_permutations(nums: List[int]) -> List[List[int]]:
  def backtrack(path, used):
    if len(path) == len(nums):
      result.append(path[:])
      return
    for i in range(len(nums)):
      if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
        continue
      used[i] = True
      path.append(nums[i])
      backtrack(path, used)
      path.pop()
      used[i] = False

  nums.sort()
  result = []
  backtrack([], [False] * len(nums))
  return result


def find_all_dates(text: str) -> List[str]:
    res = []
    lst = [r'\b\d{2}-\d{2}-\d{4}\b', r'\b\d{2}/\d{2}/\d{4}\b', r'\b\d{4}\.\d{2}\.\d{2}\b']
    for i in lst:
      res.extend(re.findall(i, text))

    return res

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)
    latitudes = []
    longitudes = []
    distances = [0]
    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:
            distance = haversine(latitudes[i - 1], longitudes[i - 1], lat, lon)
            distances.append(distance)
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]  
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]  
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def day_time_to_datetime(day, time):
    base_date = datetime(2023, 1, 2)
    day_offset = days_of_week.index(day)
    day_date = base_date + timedelta(days=day_offset)
    time_obj = datetime.strptime(time, '%H:%M:%S').time()
    return datetime.combine(day_date, time_obj)

def time_check(df: pd.DataFrame) -> pd.Series:
    
    def get_time_range(start_day, start_time, end_day, end_time):
        start_dt = day_time_to_datetime(start_day, start_time)
        end_dt = day_time_to_datetime(end_day, end_time)
        return start_dt, end_dt
    
    def check_coverage(group):
        day_intervals = {day: [] for day in days_of_week}
        
        for _, row in group.iterrows():
            start_dt, end_dt = get_time_range(row['startDay'], row['startTime'], row['endDay'], row['endTime'])
            
            while start_dt <= end_dt:
                day_str = start_dt.strftime('%A')
                day_intervals[day_str].append((start_dt.time(), end_dt.time()))
                start_dt += timedelta(days=1)
                start_dt = start_dt.replace(hour=0, minute=0, second=0)
        
        for day, intervals in day_intervals.items():
            intervals.sort()
            merged = []
            for start, end in intervals:
                if not merged or merged[-1][1] < start:
                    merged.append((start, end))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            
            if not merged or not (merged[0][0] == datetime.min.time() and merged[-1][1] == datetime.max.time()):
                return True
        
        return False
    
    result = df.groupby(['id', 'id_2'], group_keys=False).apply(check_coverage)
    
    return result


df_1 = pd.read_csv('/content/dataset-1.csv')
time_check_results_optimized = time_check_optimized(df_1)
print(time_check_results_optimized)
