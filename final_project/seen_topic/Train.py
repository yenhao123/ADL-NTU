import torch
import pandas as pd
import numpy as np
import os
import copy
import datetime
import pickle
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
pd.set_option('mode.chained_assignment', None)
torch.manual_seed(2022)

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

# torch_rehub
from torch_rechub.utils.match import gen_model_input
from torch_rechub.basic.features import SparseFeature, SequenceFeature, DenseFeature
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
    train_summary = {"user_id":[], "subgroup":[]}
    train = pd.read_csv(input_dir + "train_group.csv")

    # split subgroup
    for i, cids in enumerate(train["subgroup"]):
        
        #ids = cids.split(" ")

        if isinstance(cids, str):
            ids = cids.split(" ")
        else:
            ids = []

        for idx in ids:
            train_summary["user_id"].append(train["user_id"][i])
            train_summary["subgroup"].append(idx)


    train = pd.DataFrame(train_summary)

    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    train = train.merge(user, on=["user_id"])
    # add course features
    course = pd.read_csv(input_dir + "courses.csv")
    course = course.rename(columns = {"sub_groups" : "subgroup"})

    subgroups = pd.read_csv(input_dir + "subgroups.csv")

    # 把 train.csv subgroup 轉成 subgroup name
    subgroup_name = pd.read_csv(input_dir + "subgroups.csv")
    train.to_csv(input_dir + "train_revised.csv", index=False)
    train = pd.read_csv(input_dir + "train_revised.csv")
    subgroup_name = subgroup_name.rename(columns = {"subgroup_id" : "subgroup"})
    train = pd.merge(train,subgroup_name,on="subgroup")

    train = train.drop(columns=["subgroup"])
    train = train.rename(columns = {"subgroup_name" : "subgroup"})

    # 把 course 裡 subgroups 分割成多行，e.g, (course_1,subgroup_1)、(course_1,subgroup_2),...
    course_mapping = {"subgroup":[], "course_id":[]}
    for i, cstrs in enumerate(course["subgroup"]):
        if not pd.isna(cstrs):
            for cstr in cstrs.split(","):
                course_mapping["course_id"].append(course["course_id"][i])
                course_mapping["subgroup"].append(cstr)
        else:
            course_mapping["course_id"].append(course["course_id"][i])
            course_mapping["subgroup"].append("nan")


    # group by subgroup
    cm = pd.DataFrame(course_mapping)
    c_group = cm.groupby("subgroup",sort=False)

    train.to_csv(input_dir + "train_revised.csv", index=False)
    train = pd.read_csv(input_dir + "train_revised.csv")

    # merge each group
    x = [pd.merge(train, c_g[1], on=["subgroup"]) for c_g in c_group]
    
    train = pd.concat(x, ignore_index=True)

    course.drop(columns=["subgroup"], inplace=True)
    train = train.merge(course, on=["course_id"])

    # to csv
    #train.to_csv(summary_path,index=False)

    return train


def eval_summary(args):
      
    input_dir = args.input_dir
    val_summary = {"user_id":[], "subgroup":[]}
    val = pd.read_csv(input_dir + "val_seen_group.csv")

    # split subgroup
    for i, cids in enumerate(val["subgroup"]):
        
        #ids = cids.split(" ")

        if isinstance(cids, str):
            ids = cids.split(" ")
        else:
            ids = []

        for idx in ids:
            val_summary["user_id"].append(val["user_id"][i])
            val_summary["subgroup"].append(idx)


    val = pd.DataFrame(val_summary)

    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    val = val.merge(user, on=["user_id"])
    # add course features
    course = pd.read_csv(input_dir + "courses.csv")
    course = course.rename(columns = {"sub_groups" : "subgroup"})

    subgroups = pd.read_csv(input_dir + "subgroups.csv")

    # 把 val.csv subgroup 轉成 subgroup name
    subgroup_name = pd.read_csv(input_dir + "subgroups.csv")
    val.to_csv(input_dir + "val_revised.csv", index=False)
    val = pd.read_csv(input_dir + "val_revised.csv")
    subgroup_name = subgroup_name.rename(columns = {"subgroup_id" : "subgroup"})
    val = pd.merge(val,subgroup_name,on="subgroup")

    val = val.drop(columns=["subgroup"])
    val = val.rename(columns = {"subgroup_name" : "subgroup"})

    # 把 course 裡 subgroups 分割成多行，e.g, (course_1,subgroup_1)、(course_1,subgroup_2),...
    course_mapping = {"subgroup":[], "course_id":[]}
    for i, cstrs in enumerate(course["subgroup"]):
        if not pd.isna(cstrs):
            for cstr in cstrs.split(","):
                course_mapping["course_id"].append(course["course_id"][i])
                course_mapping["subgroup"].append(cstr)
        else:
            course_mapping["course_id"].append(course["course_id"][i])
            course_mapping["subgroup"].append("nan")


    # group by subgroup
    cm = pd.DataFrame(course_mapping)
    c_group = cm.groupby("subgroup",sort=False)

    val.to_csv(input_dir + "val_revised.csv", index=False)
    val = pd.read_csv(input_dir + "val_revised.csv")

    # merge each group
    x = [pd.merge(val, c_g[1], on=["subgroup"]) for c_g in c_group]
    
    val = pd.concat(x, ignore_index=True)

    course.drop(columns=["subgroup"], inplace=True)
    val = val.merge(course, on=["course_id"])

    # to csv
    #val.to_csv(summary_path,index=False)

    return val

def test_summary(args):
      
    input_dir = args.input_dir
    test_summary = {"user_id":[], "subgroup":[]}
    test = pd.read_csv(input_dir + "test_seen_group.csv")

    # split subgroup
    for i, cids in enumerate(test["subgroup"]):
        
        #ids = cids.split(" ")

        if isinstance(cids, str):
            ids = cids.split(" ")
        else:
            ids = []

        for idx in ids:
            test_summary["user_id"].append(test["user_id"][i])
            test_summary["subgroup"].append(idx)


    test = pd.DataFrame(test_summary)

    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    test = test.merge(user, on=["user_id"])
    # add course features
    course = pd.read_csv(input_dir + "courses.csv")
    course = course.rename(columns = {"sub_groups" : "subgroup"})

    subgroups = pd.read_csv(input_dir + "subgroups.csv")

    # 把 test.csv subgroup 轉成 subgroup name
    subgroup_name = pd.read_csv(input_dir + "subgroups.csv")
    test.to_csv(input_dir + "test_revised.csv", index=False)
    test = pd.read_csv(input_dir + "test_revised.csv")
    subgroup_name = subgroup_name.rename(columns = {"subgroup_id" : "subgroup"})
    test = pd.merge(test,subgroup_name,on="subgroup")

    test = test.drop(columns=["subgroup"])
    test = test.rename(columns = {"subgroup_name" : "subgroup"})

    # 把 course 裡 subgroups 分割成多行，e.g, (course_1,subgroup_1)、(course_1,subgroup_2),...
    course_mapping = {"subgroup":[], "course_id":[]}
    for i, cstrs in enumerate(course["subgroup"]):
        if not pd.isna(cstrs):
            for cstr in cstrs.split(","):
                course_mapping["course_id"].append(course["course_id"][i])
                course_mapping["subgroup"].append(cstr)
        else:
            course_mapping["course_id"].append(course["course_id"][i])
            course_mapping["subgroup"].append("nan")


    # group by subgroup
    cm = pd.DataFrame(course_mapping)
    c_group = cm.groupby("subgroup",sort=False)

    test.to_csv(input_dir + "test_revised.csv", index=False)
    test = pd.read_csv(input_dir + "test_revised.csv")

    # merge each group
    x = [pd.merge(test, c_g[1], on=["subgroup"]) for c_g in c_group]
    
    test = pd.concat(x, ignore_index=True)

    course.drop(columns=["subgroup"], inplace=True)
    test = test.merge(course, on=["course_id"])

    # to csv
    #test.to_csv(summary_path,index=False)
    
    return test

def summary_csv_to_pd(args):

    train_summary_path = args.input_dir + "train_summary.csv"
    train_df = train_summary(args,train_summary_path)
    eval_df = eval_summary(args)

    return train_df,eval_df


def format_to_unix_time(data):
    
    try:
        date_time_obj = datetime.datetime.strptime(data,"%Y-%m-%d %H:%M:%S.%f")
        unix_time = date_time_obj.timestamp()
    except:
        unix_time = 1594008106.226

    return unix_time

def preprocess(args,train_df,eval_df):

    user_col, item_col = "user_id", "subgroup"
    
    sparse_user_features = [
        "user_id","gender","occupation_titles",\
        "interests","recreation_names"
    ]
    
    dense_user_features = []
    
    sparse_course_features = [
        "course_id","course_name","teacher_id","teacher_intro",\
        "groups","topics","description",\
        "will_learn","required_tools",\
        "recommended_background","target_group","subgroup"
    ]
    dense_course_features = ["course_published_at_local","course_price"]

    # Nan
    users = pd.read_csv(args.input_dir + "users.csv")
    courses = pd.read_csv(args.input_dir + "courses.csv")
    #subgroups = pd.read_csv(args.input_dir + "subgroups.csv")
    courses = courses.rename(columns = {"sub_groups" : "subgroup"})
    #subgroups = subgroups.rename(columns = {"subgroup_name" : "subgroup"})
    courses = courses.fillna("nan")
    users = users.fillna("nan")
    train_df = train_df.fillna("nan")
    eval_df = eval_df.fillna("nan")

    '''
    # 處理 Dense Feature

    ## timestamp 處理

    train_df["course_published_at_local"] = train_df.apply(lambda d:format_to_unix_time(d["course_published_at_local"]),axis=1)

    eval_df["course_published_at_local"] = eval_df.apply(lambda d:format_to_unix_time(d["course_published_at_local"]),axis=1)

    ## normalize

    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(train_df[dense_course_features].values)
    train_df[dense_course_features] = x_scaled
    '''
    # 綜合 sparse features
    sparse_features = ["user_id", "subgroup"]
    for cf in sparse_course_features:
        if cf not in sparse_features:
            sparse_features.append(cf)

    for uf in sparse_user_features:
        if uf not in sparse_features:
            sparse_features.append(uf)

    # 对 SparseFeature 进行 LabelEncoding

    feature_max_idx = {}
    for feature in sparse_user_features:
        lbe = LabelEncoder()
        lbe.fit(users[feature])
        users[feature] = lbe.transform(users[feature]) + 1
        feature_max_idx[feature] = users[feature].max() + 1
        train_df[feature] = lbe.transform(train_df[feature]) + 1
        eval_df[feature] = lbe.transform(eval_df[feature]) + 1
        # user_map、item_map for mapping in inference steps
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    for feature in sparse_course_features:
        lbe = LabelEncoder()
        lbe.fit(train_df[feature])
        train_df[feature] = lbe.transform(train_df[feature]) + 1
        feature_max_idx[feature] = train_df[feature].max() + 1
        eval_df[feature] = lbe.transform(eval_df[feature]) + 1
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    np.save(args.input_dir+"raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    # 定义两个塔对应哪些特征
    user_cols = sparse_user_features
    item_cols = sparse_course_features

    train_user_profile = train_df[user_cols].drop_duplicates('user_id')
    train_item_profile = train_df[item_cols].drop_duplicates('subgroup')
    
    eval_user_profile = eval_df[user_cols].drop_duplicates('user_id')
    eval_item_profile = eval_df[item_cols].drop_duplicates('subgroup')

    #if you have run this script before and saved the preprocessed data
    process_path = args.input_dir + "data_preprocess.pickle"
    if args.load_cache:
        with open(process_path, 'rb') as f:
            new_dict = pickle.load(f)
        x_train, y_train, x_eval, y_eval = new_dict["pickle"]
    else:
        df_train, df_eval = generate_seq_feature(
            train_df,
            eval_df,
            user_col,
            item_col,
            item_attribute_cols=[],
            sample_method=1,
            mode=0,
            neg_ratio=3, #split
            min_item=0
        )

        x_train = gen_model_input(df_train, train_user_profile, user_col, \
        train_item_profile, item_col, seq_max_len=50)
        y_train = x_train["label"]

        x_eval = gen_model_input(df_eval, eval_user_profile, user_col, \
        eval_item_profile, item_col, seq_max_len=50)
        y_eval = x_eval["label"]

        mydict = {"pickle":np.array((x_train, y_train, x_eval, y_eval))}
        with open(process_path, 'wb') as f:
            pickle.dump(mydict, f, protocol=4)


    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], \
        embed_dim=16) for feature_name in user_cols
    ]
    user_features += [
        SequenceFeature("hist_subgroup",
                        vocab_size=feature_max_idx["subgroup"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="subgroup")
    ]

    item_features = [
            SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], \
            embed_dim=16) for feature_name in sparse_course_features
    ]
    
    
    val_user = x_eval
    
    all_item = df_to_dict(train_item_profile)

    return user_features, item_features, x_train, y_train, val_user, all_item

def inference(args, user_features, item_features, x_train, y_train, val_user, all_item):
    
    # 根据之前处理的数据拿到Dataloader
    dg = MatchDataGenerator(x=x_train, y=y_train)

    # 產生 train、eval 的 dataloader
    train_dl, eval_dl, item_dl = dg.generate_dataloader(val_user, all_item, batch_size=args.batch_size, num_workers=args.num_workers)

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

    model.load_state_dict(torch.load("output/model.pth"))

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

    # for eval
    save_dir = "./output/"
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=eval_dl, model_path=args.save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.save_dir)
    raw_id_maps_path = "./input/hahow/data/raw_id_maps.npy"
    match_evaluation(args,user_embedding, item_embedding, val_user, all_item, raw_id_maps=raw_id_maps_path, topk=50)


def train(args, user_features, item_features, x_train, y_train, val_user, all_item):
    
    # 根据之前处理的数据拿到Dataloader
    dg = MatchDataGenerator(x=x_train, y=y_train)

    # 產生 train、eval 的 dataloader
    train_dl, eval_dl, item_dl = dg.generate_dataloader(val_user, all_item, batch_size=args.batch_size, num_workers=args.num_workers)

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
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=eval_dl, model_path=args.save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.save_dir)
    raw_id_maps_path = args.input_dir + "raw_id_maps.npy"
    match_evaluation(args,user_embedding, item_embedding, val_user, all_item, raw_id_maps=raw_id_maps_path, topk=50)

def run(args):

    train_df,eval_df = summary_csv_to_pd(args)

    user_features, item_features, x_train, y_train, val_user, all_item = preprocess(args,train_df,eval_df)

    if not args.test:
        train(args,user_features, item_features, x_train, y_train, val_user, all_item)
    else:
        inference(args,user_features, item_features, x_train, y_train, val_user, all_item)

if __name__ == "__main__":
    args = set_arg()
    run(args)