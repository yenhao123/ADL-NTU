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
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import df_to_dict
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

# own package
from Set_Arg import set_arg
from utils import match_evaluation

def summary_csv_to_pd(args,summary_path):

    input_dir = args.input_dir
    train_summary = {"user_id":[],"course_id":[]}
    train = pd.read_csv(input_dir+"train.csv")
    
    train_columns_beforeadd = train_summary.copy().keys()
    
    # process train.csv
    ## split course_id
    for i,cids in enumerate(train["course_id"]):
        ids = cids.split(" ")
        for idx in ids: 
            train_summary["user_id"].append(train["user_id"][i])
            train_summary["course_id"].append(idx)

    # add course features
    course = pd.read_csv(input_dir+"courses.csv")
    
    for col in course.columns:
        if col not in train_columns_beforeadd:
            train_summary.update({col:[]})
    
    print(train_columns_beforeadd)
    ## by order of selled course in train.csv, add corresponding value in course.csv
    for cid in train_summary["course_id"]:
        mask = course["course_id"].str.match(pat = cid)
        selectedRows = course[mask].drop(columns=["course_id"])

        for col in selectedRows.columns:
            if col not in train_columns_beforeadd:
                train_summary[col].append(selectedRows[col].values[0])

    ## add column
    train_columns_beforeadd = train_summary.copy().keys()
    
    # add chapter features
    #chapter = pd.read_csv(input_dir+"course_chapter_items.csv")

    # add user features
    user = pd.read_csv(input_dir+"users.csv")
    for col in user.columns:
        if col not in train_columns_beforeadd:
            train_summary.update({col:[]})
    
    ## by order of user in train.csv, add corresponding value in user.csv
    cnt = 0
    for uid in train_summary["user_id"]:
        if cnt % 1000 == 0:
            print(uid + str(cnt))
        mask = user["user_id"].str.match(pat = uid)
        selectedRows = user[mask].drop(columns=["user_id"])

        for col in selectedRows.columns:
            if col not in train_columns_beforeadd:
                train_summary[col].append(selectedRows[col].values[0])
        cnt += 1

    ## add column
    train_columns_beforeadd = train_summary.keys()

    # to csv
    train_df = pd.DataFrame(train_summary)

    ## timestamp 處理
    formatToUnixTime(train_df["course_published_at_local"])

    train_df.to_csv(summary_path,index=False)

    return train_df

def formatToUnixTime(time):

    timestamps = []
    for i,t in enumerate(time):
        try:
            date_time_obj = datetime.datetime.strptime(t,"%Y-%m-%d %H:%M:%S.%f")
            time[i] = date_time_obj.timestamp()
        except:
            time[i] = 1594008106.226

def preprocess(args,data):
    
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

    '''
    # timestamp 處理
    formatToUnixTime(data["course_published_at_local"])
    '''

    # summary sparse features
    sparse_features = ["user_id", "course_id"]
    for cf in sparse_course_features:
        if cf not in sparse_features:
            sparse_features.append(cf)
    
    for uf in sparse_user_features:
        if uf not in sparse_features:
            sparse_features.append(uf)

    # 对SparseFeature进行LabelEncoding
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        # user_map、item_map for mapping in inference steps
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    np.save(args.input_dir+"raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    # 定义两个塔对应哪些特征
    user_cols = sparse_user_features
    item_cols = sparse_course_features
    user_profile = data[user_cols].drop_duplicates('user_id')
    item_profile = data[item_cols].drop_duplicates('course_id')
    
    #if you have run this script before and saved the preprocessed data
    processed_data_path = args.input_dir + "data_preprocess.npy"
    if args.load_cache:  
        x_train, y_train, x_test, y_test = np.load(processed_data_path, allow_pickle=True)
    else:
        df_train, df_test = generate_seq_feature_match(
            data,
            user_col,
            item_col,
            time_col="course_published_at_local",
            item_attribute_cols=[],
            sample_method=1,
            mode=0,
            neg_ratio=3, #split
            min_item=0
        )
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_train = x_train["label"]
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_test = x_test["label"]
        np.save(processed_data_path, np.array((x_train, y_train, x_test, y_test), dtype=object))

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
    
    all_item = df_to_dict(item_profile)
    test_user = x_test

    return user_features, item_features, x_train, y_train, all_item, test_user

def train(args, user_features, item_features, x_train, y_train, all_item, test_user):
    
    # 根据之前处理的数据拿到Dataloader
    dg = MatchDataGenerator(x=x_train, y=y_train)

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

    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=args.batch_size)
    trainer.fit(train_dl)

    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=args.save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=args.save_dir)
    torch.save(user_embedding.data.cpu(), args.save_dir + "user_embedding.pth")
    torch.save(item_embedding.data.cpu(), args.save_dir + "item_embedding.pth")
    raw_id_maps_path = args.input_dir + "raw_id_maps.npy"
    match_evaluation(user_embedding, item_embedding, test_user, all_item, raw_id_maps_path)

def run(args):

    summary_path = args.input_dir + "summary.csv"
    if not os.path.exists(summary_path):
        train_df = summary_csv_to_pd(args,summary_path)
    else:
        train_df = pd.read_csv(summary_path)

    user_features, item_features, x_train, y_train, all_item, test_user = preprocess(args,train_df)
    
    train(args,user_features, item_features, x_train, y_train, all_item, test_user)



if __name__ == "__main__":
    args = set_arg()
    run(args)