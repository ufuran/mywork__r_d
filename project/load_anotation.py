import json
import pandas as pd

def save_annotations(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['file_size', 'file_attributes', 'region_count', 'region_id', 'region_attributes'], axis=1)
    df = df.rename(columns={'#filename': 'image_name'})

    def convert_region_shape_attributes(row):
        _l = json.loads(row['region_shape_attributes'])
        return pd.Series({'geometry': [(_l['x'], _l['y']), (_l['x'] + _l['width'], _l['y']), (_l['x'] + _l['width'], _l['y'] + _l['height']), (_l['x'], _l['y'] + _l['height']), (_l['x'], _l['y'])], 'class': 'screw', 'x': _l['x'], 'y': _l['y'], 'w': _l['width'], 'h': _l['height']})

    df = df.join(df.apply(convert_region_shape_attributes, axis=1))
    df = df.drop(['region_shape_attributes'], axis=1)
    df = df.rename(columns={'image_name': 'image_id'})

    df.to_csv('anotation_all_new.csv')