# -*- encoding: utf-8 -*-
# @Time     :   2023/07/02 21:25:46
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   None

import numpy as np

dictionaries =  np.load('./physicochemical_properties/dictionaries_for_grouping.npy', allow_pickle=True)

def replace(content, d):
    r = list(content)
    for i in range(len(content)):
        for j in d:
            r[i] = r[i].replace(j, str(d[j]))
    return r


if __name__ == "__main__":
    
    pass