import pandas as pd
import os
import datetime
import numpy as np
import json
import argparse
from json import JSONEncoder

def setArg():
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--inputDir", type=str,default="./input/hahow")
    
    args = parser.parse_args()
    return args


def countUnique(df):
    s_colnames =  ["course_name","teacher_id","teacher_intro",\
    "groups","sub_groups","topics","description","will_learn","required_tools",\
    "recommended_background","target_group"]
    print(len(s_colnames))
    for col in s_colnames:
        print("col:{};unique_value:{}".format(col,str(len(pd.unique(df[col])))))
    


if __name__ == "__main__":
    args = setArg()
    summaryPath = args.inputDir+"/trainSummary.csv"
    if not os.path.exists(summaryPath):
        df = summaryCSV()
    else:
        df = pd.read_csv(summaryPath) 

    countUnique(df)