# Computer Vision application

## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Setup](#setup)
4. [Modules](#modules)
5. [Authors](#authors)
6. [Environment](#environment)


### General Info
This project is based on a set of images that represent some possible electrical sources located in a specific area of north Italy. 
The goal is inclined to build up a tool that finds hot spots in IR images by exploiting Computer Vision techniques. 
The request consists into apply a Computer Vision application that allows to localize those trellises with an outlier temperature.
Once that anomalies are located they are showed to domain specialists in such a way that they can be easily fixed. 


### Technologies

A list of technologies used within the project:
* [python](https://.com): Version 3.7.0
* [pandas](https://.com): Version 1.1.5 
* [numpy](https://.com): Version 1.19.5
* [opencv-python](https://.com): Version 4.5.5.64
* [pybase64](https://.com): Version 1.2.2
* [utm](https://.com): Version 0.7.0
* [pathlib](https://.com): Version 1.0.1
* [boto3](https://.com): Version 1.21.45
* [CouchDB](https://.com): Version 1.2


### Setup
Follow the below steps to run the project.

* Clone the Git repository
```
$ git clone https://github.com/DoubleElle1993/Computer-vision-algorithm.git
```
* Achive the local path
```
$ cd local-path\Irb-Project\Tralicci-IRB
```
* Install requirements.txt
```
$ pip install -r requirements.txt
```
* Run analysis module 
```
$ run analysis.py
```


### Modules

This project is composed of the following modules. 

| Name | Description |
| --   | ----------- |
| analysis.py| This module provides analytics functions that are used within the script     |
| utils.py| This module provides utility functions that are used within the script    |
| databaseirb.py| This module provides a function to originate a CouchDB    |
| config.py| This module provides variables and access keys that are used within the script    |



