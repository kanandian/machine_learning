import numpy as np
import matplotlib.pyplot as plt
import csv


def prehand_data(data):
    res = []
    timepoint = {}
    temp = []
    pre_key = 0
    for item in data:
        key = item.get('日期')
        if pre_key == 0:
            pre_key = key
        if key == pre_key:
            temp.append(item)
        else:
            for h in range(24):
                for i in range(len(temp)):
                    p_key = temp[i].get('测项')
                    p_value = temp[i].get(str(h))
                    timepoint[p_key] = p_value
                res.append(timepoint)

            pre_key = key
            temp = []
            temp.append(item)
    for h in range(24):
        for i in range(len(temp)):
            p_key = temp[i].get('测项')
            p_value = temp[i].get(str(h))
            timepoint[p_key] = p_value
        res.append(timepoint)

    return res

def main():
    # 参数
    p_amb_temp = 10
    p_ch4 = 10
    p_co = 10
    p_nmhc = 10
    p_no = 10
    p_no2 = 10
    p_nox = 10
    p_o3 = 10
    p_pm10 = 10
    p_pm25 = 10
    p_rainfall = 10
    p_rh = 10
    p_so2 = 10
    p_thc = 10
    p_wd_hr = 10
    p_wind_dirc = 10
    p_wind_speed = 10
    p_ws_hr = 10

    # settings
    iteration = 300000
    lr = 0.0001

    train_data = []

    with open('PM2.5_DATA/train.csv', 'r') as f:
        dict_reader = csv.DictReader(f)

        for dict in dict_reader:
            train_data.append(dict)

    train_data = prehand_data(train_data)
    print(train_data[len(train_data)-1])




    # for item in train_data:
    #     print(item)
    #
    # for i in range(iteration):
    #     for item in train_data:
    #         pass


if __name__ == '__main__':
    main()

class Paras():
    def __init__(self):
        self.amb_temp = 10
        self.ch4 = 10
        self.co = 10
        self.nmhc = 10
        self.no = 10
        self.no2 = 10
        self.nox = 10
        self.o3 = 10
        self.pm10 = 10
        self.pm25 = 10
        self.rainfall = 10
        self.rh = 10
        self.so2 = 10
        self.thc = 10
        self.wd_hr = 10
        self.wind_dirc = 10
        self.wind_speed = 10
        self.ws_hr = 10