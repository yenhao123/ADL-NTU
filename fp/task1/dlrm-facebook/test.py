import pandas as pd
import os
import datetime
import numpy as np
import json
import argparse
from json import JSONEncoder

s_colnames =  ["course_name","teacher_id","teacher_intro",\
    "groups","sub_groups","topics","description","will_learn","required_tools",\
    "recommended_background","target_group"]

s_features = []
users = 139608
for i,s_colname in enumerate(s_colnames):
    s_feature = np.full((users,i+1),i)
    if s_colname == s_colnames[0]:
        s_feature = s_feature.reshape(s_feature.shape[0],1,s_feature.shape[1])
        s_features = s_feature.tolist()
    else:
        s_feature = s_feature.tolist()
        for j in range(users):
            s_features[j].append(s_feature[j])
    
print(s_features[0])
print(len(s_features))