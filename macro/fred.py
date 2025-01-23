import json

import pandas as pd
import requests


OBSERVATIONS_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations?"


def create_arg_string(arg_dict):
    args = ["=".join([k, v]) for k, v in arg_dict.items()]
    return "&".join(args)


def query_observations(arg_dict, api_key):
    arg_string = create_arg_string(arg_dict)
    rate_url = f"{OBSERVATIONS_ENDPOINT}{arg_string}&api_key={api_key}&file_type=json"
    return requests.get(rate_url)


def observations_to_table(observations_response):
    observations_json = json.loads(observations_response.content)
    return pd.DataFrame.from_dict(observations_json['observations'])


def drop_past_date(table):
    table = table.copy()
    table.sort_values(by=['date', 'realtime_start'], inplace=True)
    # Keep only the realtime_start with the latest date
    table.drop_duplicates(subset=['date'], keep='last', inplace=True)
    table['date'] = pd.to_datetime(table['date'])
    return table


def query_to_table(arg_dict, api_key, deduplicate=True):
    response = query_observations(arg_dict, api_key)
    table = observations_to_table(response)
    if deduplicate:
        return drop_past_date(table)
    return table
