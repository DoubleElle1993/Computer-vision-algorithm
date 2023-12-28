import os
import pandas as pd
import boto3
from datetime import date

# MAIN FUNCTION
cwd = os.getcwd()
bucket_name = "" #bucket name
keys = pd.read_csv(os.path.join(cwd, os.path.join(cwd, '.csv')))  #csv file with access keys
aws_access_key_id, aws_secret_access_key = keys['access_key'][0], keys['secret_access_key'][0]
resource = boto3.resource(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='eu-west-1'
)
bucket = resource.Bucket(bucket_name)
client = boto3.client('s3', region_name='eu-west-1')
paginator = client.get_paginator('list_objects')
page_iterator = paginator.paginate(Bucket='') #bucket name 


# CREATE_JSON FUNCTION
result_dict = {'name': None, 'link': None, 'date': None, 'day': None,
               'coordinates': dict(), 'season': None, 'threshold': None, 'max_temp': None,
               'min_temp': None, 'unit': None, 'analysis': dict()}
winter_threshold = 40
spring_threshold = 40
summer_threshold = 40
autumn_threshold = 40
keys_to_delete = ['image', 'image_bb', 'height', 'width',
                  'x', 'y']


# MAX_TEMP FUNCTION
analysis_dict = {'defect_bool': None, 'defect_image_name': None, 'image': None, 'delta': None,
                 'image_bb': None, 'height': list(), 'width': list(),
                 'x': list(), 'y': list()}
kernelSize = 15
opIterations = 6


# CREATE_TXT FUNCTION
prefix_list = [''] #prefix list names 
columns_list = ['x', 'y', 'h',
                'date', 'time', 'folder',
                'empty', 'unknown', 'datetime']


# SCALER FUNCTION
output_values = (255, 0)


# GET_SEASON FUNCTION
Y = 2000
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]


# CREATE_CSV_OUTPUT FUNCTION
csv_output_columns = ['Name', 'Link', 'Date', 'Day',
                      'Season', 'Threshold', 'Max_temp', 'Min_temp',
                      'Unit', 'Utm', 'Lat-Lon', 'Defect_bool',
                      'Defect_Image_Name', 'Delta']

readable_columns_csv = ['Name', 'Link', 'Date', 'Day',
                        'Season', 'Threshold', 'Max_temp', 'Min_temp',
                        'Delta', 'Unit', 'Utm', 'Lat-Lon', 'Defect_bool',
                        'Defect_Image_Name']

