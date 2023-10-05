import torch
import pandas as pd
import numpy as np
import os
import copy
import datetime
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
pd.set_option('mode.chained_assignment', None)
torch.manual_seed(2022)

from sklearn.preprocessing import LabelEncoder

# torch_rehub
from torch_rechub.utils.match import gen_model_input
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import df_to_dict
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

# own package
from set_arg import set_arg
from utils import match_evaluation, generate_seq_feature
import tqdm

def train_summary(args,summary_path):
    
    input_dir = args.input_dir
    train_summary = {"user_id":[],"course_id":[]}
    train = pd.read_csv(input_dir+"train.csv")

    # process train.csv
    ## split course_id
    for i,cids in enumerate(train["course_id"]):
        ids = cids.split(" ")
        for idx in ids: 
            train_summary["user_id"].append(train["user_id"][i])
            train_summary["course_id"].append(idx)

    train = pd.DataFrame(train_summary)

    # add user features
    user = pd.read_csv(input_dir+"users.csv")
    train = train.merge(user,on=["user_id"])

    # add course features
    course = pd.read_csv(input_dir+"courses.csv")
    train = train.merge(course,on=["course_id"])

    # add chapter features
    #chapter = pd.read_csv(input_dir+"course_chapter_items.csv")

    # to csv
    train.to_csv(summary_path,index=False)

    return train


def eval_summary(args,summary_path):
    
    input_dir = args.input_dir
    val_summary = {"user_id":[],"course_id":[]}
    val = pd.read_csv(input_dir+"val_seen.csv")

    # process train.csv
    ## split course_id
    for i,cids in enumerate(val["course_id"]):
        ids = cids.split(" ")
        for idx in ids: 
            val_summary["user_id"].append(val["user_id"][i])
            val_summary["course_id"].append(idx)

    val = pd.DataFrame(val_summary)

    # add user features
    user = pd.read_csv(input_dir+"users.csv")
    val = val.merge(user,on=["user_id"])

    # add course features
    course = pd.read_csv(input_dir+"courses.csv")
    val = val.merge(course,on=["course_id"])
    val.to_csv(summary_path)
    
    return val

def test_summary(args):
    
    input_dir = args.input_dir
    test_summary = {"user_id":[],"course_id":[]}
    test = pd.read_csv(input_dir+"test_seen.csv")

    # process train.csv
    ## split course_id
    for i,cids in enumerate(test["course_id"]):
        ids = cids.split(" ")
        for idx in ids: 
            test_summary["user_id"].append(test["user_id"][i])
            test_summary["course_id"].append(idx)

    test = pd.DataFrame(test_summary)

    # add user features
    user = pd.read_csv(input_dir+"users.csv")
    test = test.merge(user,on=["user_id"])

    # add course features
    course = pd.read_csv(input_dir+"courses.csv")
    test = test.merge(course,on=["course_id"])
    
    return test

def summary_csv_to_pd(args):

    train_summary_path = args.input_dir + "train_summary.csv"
    if not os.path.exists(train_summary_path):
        train_df = train_summary(args,train_summary_path)
    else:
        train_df = pd.read_csv(train_summary_path)

    eval_summary_path = args.input_dir + "eval_summary.csv"
    if not os.path.exists(eval_summary_path):
        eval_df = eval_summary(args,eval_summary_path)
    else:
        eval_df = pd.read_csv(eval_summary_path)

    test_df = test_summary(args)

    return train_df,eval_df,test_df


def format_to_unix_time(data):
    
    try:
        date_time_obj = datetime.datetime.strptime(data,"%Y-%m-%d %H:%M:%S.%f")
        unix_time = date_time_obj.timestamp()
    except:
        unix_time = 1594008106.226

    return unix_time

def preprocess(args,train_df,eval_df,test_df):

    user_col, item_col = "user_id", "course_id"
    
    sparse_user_features = [
        "user_id","gender","occupation_titles",\
        "interests","recreation_names"
    ]


    dense_user_features = []
    
    sparse_course_features = [
        "course_id","course_name","teacher_id","teacher_intro",\
        "groups","sub_groups","topics","description",\
        "will_learn","required_tools",\
        "recommended_background","target_group"
    ]
    dense_course_features = ["course_published_at_local","course_price"]

    # timestamp 處理
    train_df["unix_time"] = train_df.apply(lambda d:format_to_unix_time(d["course_published_at_local"]),axis=1)
    train_df.drop(["course_published_at_local"],axis=1,inplace=True)

    test_df["unix_time"] = test_df.apply(lambda d:format_to_unix_time(d["course_published_at_local"]),axis=1)
    test_df.drop(["course_published_at_local"],axis=1,inplace=True)

    # 綜合 sparse features
    sparse_features = ["user_id", "course_id"]
    for cf in sparse_course_features:
        if cf not in sparse_features:
            sparse_features.append(cf)

    for uf in sparse_user_features:
        if uf not in sparse_features:
            sparse_features.append(uf)

    # 对 SparseFeature 进行 LabelEncoding
    users = pd.read_csv(args.input_dir + "users.csv")

    feature_max_idx = {}
    for feature in sparse_user_features:
        lbe = LabelEncoder()
        lbe.fit(users[feature])
        users[feature] = lbe.transform(users[feature]) + 1
        feature_max_idx[feature] = users[feature].max() + 1
        train_df[feature] = lbe.transform(train_df[feature]) + 1
        test_df[feature] = lbe.transform(test_df[feature]) + 1
        # user_map、item_map for mapping in inference steps
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    courses = pd.read_csv(args.input_dir + "courses.csv")
    for feature in sparse_course_features:
        lbe = LabelEncoder()
        lbe.fit(courses[feature])
        courses[feature] = lbe.transform(courses[feature]) + 1
        feature_max_idx[feature] = courses[feature].max() + 1
        train_df[feature] = lbe.transform(train_df[feature]) + 1
        test_df[feature] = lbe.transform(test_df[feature]) + 1
        # user_map、item_map for mapping in inference steps
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    np.save(args.input_dir+"raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    # 定义两个塔对应哪些特征
    user_cols = sparse_user_features
    item_cols = sparse_course_features

    train_user_profile = train_df[user_cols].drop_duplicates('user_id')
    train_item_profile = train_df[item_cols].drop_duplicates('course_id')

    test_user_profile = test_df[user_cols].drop_duplicates('user_id')
    test_item_profile = test_df[item_cols].drop_duplicates('course_id')

    #if you have run this script before and saved the preprocessed data
    df_train, df_test = generate_seq_feature(
        train_df,
        test_df,
        user_col,
        item_col,
        item_attribute_cols=[],
        sample_method=1,
        mode=0,
        neg_ratio=3, #split
        min_item=0
    )

    x_train = gen_model_input(df_train, train_user_profile, user_col, train_item_profile, item_col, seq_max_len=50)
    y_train = x_train["label"]

    x_test = gen_model_input(df_test, test_user_profile, user_col, test_item_profile, item_col, seq_max_len=50)
    y_test = x_test["label"]

    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in user_cols
    ]
    user_features += [
        SequenceFeature("hist_course_id",
                        vocab_size=feature_max_idx["course_id"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="course_id")
    ]

    item_features = [
            SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in item_cols
    ]
    
    courses_profile = courses[item_cols].drop_duplicates('course_id')
    all_item = df_to_dict(courses_profile)
    y_test = df_to_dict(test_item_profile)

    return user_features, item_features, x_train, y_train, x_test, y_test, all_item

def train(args, user_features, item_features, x_train, y_train, x_test, y_test, all_item):

    # 根据之前处理的数据拿到Dataloader
    dg = MatchDataGenerator(x=x_train, y=y_train)

    # 產生 train、eval 的 dataloader
    train_dl, test_dl, item_dl = dg.generate_dataloader(x_test, all_item, batch_size=args.batch_size, num_workers=args.num_workers)

    model = DSSM(
        user_features,
        item_features,
        temperature=0.02,
        user_params={
            "dims": [256, 128, 64],
            "activation": 'prelu',  # important!!
        },
        item_params={
            "dims": [256, 128, 64],
            "activation": 'prelu',  # important!!
         }
    )

    trainer = MatchTrainer(
        model,
        mode=0,
        optimizer_params={
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay
        },
        n_epoch=args.epoch,
        device=args.device,
        model_path=args.save_dir
    )

    trainer.fit(train_dl)

    # for eval
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=args.save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.save_dir)
    raw_id_maps_path = args.input_dir + "raw_id_maps.npy"
    match_evaluation(args,user_embedding, item_embedding, x_test, all_item, raw_id_maps_path,topk=50)

def run(args):

    train_df,eval_df,test_df = summary_csv_to_pd(args)

    user_features, item_features, x_train, y_train, x_test, y_test, all_item = preprocess(args,train_df,eval_df,test_df)

    train(args,user_features, item_features, x_train, y_train, x_test, y_test, all_item)



if __name__ == "__main__":
    args = set_arg()
    run(args)