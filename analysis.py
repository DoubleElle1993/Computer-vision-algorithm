import json
import numpy as np
import cv2
import base64
import config
import copy
import io
import yappi
import pandas as pd
import utm
import utils
import databaseirb
import os
from pathlib import Path
from datetime import datetime
from io import StringIO


def scaler(array, threshold, output_values=config.output_values):
    """
    This function applies a normalization formula to temperature arrays and threshold.
    It returns the scaled variables. The goal of this function is to obtain a set of values
    that fall in a range of [0,250].
    """

    input_min = min(array.flatten())
    input_max = max(array.flatten())
    output_min = min(output_values)
    output_max = max(output_values)
    function = lambda x: ((x - input_min)/(input_max - input_min)) *\
                         (output_max - output_min) + output_min
    scaled_array = function(array.flatten())
    scaled_array = np.reshape(scaled_array, (768, 1024))
    if input_max <= threshold:
        scaled_threshold = None
    else:
        scaled_threshold = int(((threshold - input_min)/(input_max - input_min)) *\
                               (output_max - output_min) + output_min)

    delta = np.round(input_max - input_min, 2)

    return scaled_array, scaled_threshold, delta


def temp_scale_legend_RGB(img, max, min):
    """
    This function enables to add a scale into the images.
    """

    white_arr = np.full((img.shape[0] + 40, img.shape[1] + 90, 3), 255, dtype=int)
    heatmap = np.repeat(np.arange(0, 256).reshape(256, 1) * np.ones((256, 40)), repeats=3, axis=0)[::-1]
    white_arr[19:787, 19:1043, :] = img
    heatmap_1 = heatmap[:, :, np.newaxis]
    white_arr[19:787, 1053:1093, :] = heatmap_1
    white_arr = white_arr.astype(np.uint8)
    #white_arr = np.kron(white_arr, np.ones((4, 4, 1)))
    white_arr = cv2.putText(white_arr, str(np.round(max, 2)) + ' C', (1055, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    white_arr = cv2.putText(white_arr, str(np.round(min, 2)) + ' C', (1055, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return white_arr


def base64_encoder(img):
    """
    This function allows to convert a numpy array into base64 format.
    """

    _, im_arr = cv2.imencode('.jpg', img)
    im_byte = im_arr.tobytes()
    im_b64 = base64.b64encode(im_byte).decode()

    return im_b64


def create_txt(bucket=config.bucket, prefix=config.prefix_list, columns=config.columns_list):
    """
    This function is strictly related to txt files stored in the Laser folder of the HardDisk.
    It's useful to get the coordinates of the drone path. 
    """

    txt_df = pd.DataFrame()
    for pref in prefix:
        prefix_objs = bucket.objects.filter(Prefix=pref)
        for obj in prefix_objs:
            key = obj.key
            if key.endswith('.txt'):
                body = obj.get()['Body'].read()
                df = pd.read_csv(io.BytesIO(body), sep=';', decimal=',', header=None,
                                 names=['col' + str(x) for x in range(16)])
                df = df.dropna(axis=1, how='all')
                txt_df = pd.concat([txt_df, df], ignore_index=True)

    txt_df['date'] = pd.to_datetime(txt_df[['col3', 'col4']].apply(' '.join, 1),
                                      format='%d/%m/%Y %H.%M.%S')
    txt_df.columns = columns
    txt_df['lat'] = utm.to_latlon(txt_df['x'], txt_df['y'], 32, 'T')[0]
    txt_df['lon'] = utm.to_latlon(txt_df['x'], txt_df['y'], 32, 'T')[1]
    txt_df['x'] = [np.round(float(i), 2) for i in txt_df['x']]
    txt_df['y'] = [np.round(float(i), 2) for i in txt_df['y']]
    txt_df.to_csv('finaltxt.csv')

    return txt_df


def create_json(csv, document_path, df_coordinates=create_txt()):
    """
    The goal of this function is to create an informative dictionary containing all information
    about the analysis.
    """
    output = config.result_dict
    if csv.startswith('1024'):
        df = pd.read_csv(StringIO(csv), sep=';', decimal=',', header=None, skiprows=1)
        df = df.dropna(axis=1, how='all')
    elif csv.startswith('[Settings]'):
        df = pd.read_csv(StringIO(csv), sep=';', decimal=',', header=None, skiprows=9)
        df = df.dropna(axis=1, how='all')
    output['name'] = os.path.basename(document_path)
    output['link'] = document_path
    if output['name'].endswith('.irb.csv'):
        date = datetime.strptime(output['name'], 'Record_%Y-%m-%d_%H-%M-%S_%f.irb.csv')
    else:
        date = datetime.strptime(output['name'], 'Record_%Y-%m-%d_%H-%M-%S_%f.csv')
    output['date'] = json.dumps(date, default=str)
    output['season'] = utils.get_season(date)
    if output['season'] == 'winter':
        output['threshold'] = config.winter_threshold
    elif output['season'] == 'spring':
        output['threshold'] = config.spring_threshold
    elif output['season'] == 'summer':
        output['threshold'] = config.summer_threshold
    elif output['season'] == 'autumn':
        output['threshold'] = config.autumn_threshold
    output['day'] = str(date.strftime('%d %B, %Y - %H:%M:%S'))
    if pd.to_datetime(date.replace(microsecond=0)) in df_coordinates["datetime"].values:
        output['coordinates']['utm'] = [df_coordinates.loc[df_coordinates["datetime"] == date.replace(microsecond=0),
                                                           ["x"]].values.item(),
                                        df_coordinates.loc[df_coordinates["datetime"] == date.replace(microsecond=0),
                                                           ["y"]].values.item()]
        output['coordinates']['lat-lon'] = [df_coordinates.loc[df_coordinates["datetime"] == date.replace(microsecond=0),
                                                               ["lat"]].values.item(),
                                            df_coordinates.loc[df_coordinates["datetime"] == date.replace(microsecond=0),
                                                               ["lon"]].values.item()]
    else:
        output['coordinates']['utm'] = "Unable to associate the timestamp with the coordinates"
        output['coordinates']['lat-lon'] = "Unable to associate the timestamp with the coordinates"
    output['max_temp'] = df.values.max()
    output['min_temp'] = df.values.min()
    output['unit'] = 'Celsius'
    output['analysis'] = max_temp(csv, output['threshold'], output['name'])
    dict_to_csv = copy.deepcopy(output)
    for i in config.keys_to_delete:
        dict_to_csv = utils.remove_key(dict_to_csv, 'analysis', i)
    utils.create_csv_output(dict_to_csv)

    return output


def max_temp(csv, threshold, name):
    """
    This function represents the whole analysis developed to find the anomalies.
    """

    result = config.analysis_dict
    result['defect_image_name'] = None
    if csv.startswith('1024'):
        df = pd.read_csv(StringIO(csv), sep=';', decimal=',', header=None, skiprows=1)
    elif csv.startswith('[Settings]'):
        df = pd.read_csv(StringIO(csv), sep=';', decimal=',', header=None, skiprows=9)
    df = df.dropna(axis=1, how='all')
    df_array = df.to_numpy(dtype=np.float64)
    scaled_df_array, scaled_threshold, delta = scaler(df_array, threshold)
    result['delta'] = delta
    if not scaled_threshold:
        result['defect_bool'] = False
    else:
        result['defect_bool'] = True
    scaled_df_array = scaled_df_array.astype(int)
    scaled_df_array = scaled_df_array.astype(np.uint8)
    scaled_image = cv2.resize(scaled_df_array, (1024, 768))
    scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
    #yappi.get_clock_type()
    #yappi.start()
    scaled_image = temp_scale_legend_RGB(scaled_image, df_array.max(), df_array.min())
    #yappi.stop()
    #yappi.get_func_stats().print_all()
    #yappi.get_thread_stats().print_all()
    image = base64_encoder(scaled_image)
    result['image'] = image
    if result['defect_bool']:
        print('Anomalia trovata')
        result['defect_image_name'] = Path(name).stem + '.jpg'
        lower = np.array([scaled_threshold], dtype='uint8')
        upper = np.array([255], dtype='uint8')
        mask = cv2.inRange(scaled_df_array, lower, upper)
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.kernelSize, config.kernelSize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morphKernel, None, None, config.opIterations, cv2.BORDER_REFLECT101)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        if contours:
            scaled_image = cv2.resize(scaled_df_array, (1024, 768))
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
            for i in contours:
                x, y, w, h = cv2.boundingRect(i)
                result['x'].append(cv2.boundingRect(i)[0])
                result['y'].append(cv2.boundingRect(i)[1])
                result['width'].append(cv2.boundingRect(i)[2])
                result['height'].append(cv2.boundingRect(i)[3])
                cv2.rectangle(scaled_image, (x, y), (x + w + 2, y + h + 2), (0, 0, 255), 2)
            scaled_image = temp_scale_legend_RGB(scaled_image, df_array.max(), df_array.min())
            image_bb = base64_encoder(scaled_image)
            result['image_bb'] = image_bb
            cv2.imwrite('Output/DefectImages/' + result['defect_image_name'], scaled_image)
            print('Immagine con anomalia salvata')

    return result


def main(page_iterator=config.page_iterator):
    """
    This is the main function.
    """
    for page in page_iterator:
        for key in page['Contents']:
            document_path = key['Key']
            if document_path.endswith('.csv'):
                obj = config.client.get_object(Bucket=config.bucket_name, Key=key['Key'])
                csv = obj['Body'].read().decode('iso8859_15')
                result_dict = create_json(csv, document_path)

                database = databaseirb.couchDBconnection()
                database.create(result_dict)
                print('The file is saved in CouchDB')

    return 0


if __name__ == "__main__":
    main()

