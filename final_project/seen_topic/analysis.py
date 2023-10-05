import numpy as np
import torch

def match_evaluation(user_embedding, item_embedding, test_user, all_item, 
                     raw_id_maps, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map, subgroup_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    preds,users = [],[]
    user_subgroups = []
    for user_id, user_emb in zip(test_user[user_col], user_embedding):

        if len(user_emb.shape)==2:
            #多兴趣召回
            items_idx = []
            items_scores = []
            for i in range(user_emb.shape[0]):
                temp_items_idx, temp_items_scores = annoy.query(v=user_emb[i], n=topk)  # the index of topk match items
                items_idx += temp_items_idx
                items_scores += temp_items_scores
            temp_df = pd.DataFrame()
            temp_df['item'] = items_idx
            temp_df['score'] = items_scores
            temp_df = temp_df.sort_values(by='score', ascending=True)
            temp_df = temp_df.drop_duplicates(subset=['item'], keep='first', inplace=False)
            recall_item_list = temp_df['item'][:topk].values
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][recall_item_list])
        else:
            #普通召回
            items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])
            user_subgroup = []
            for cid in items_idx:
                subgroup = subgroup_map[all_item["sub_groups"][cid]]
                user_subgroup.append(subgroup)

        preds.append(match_res[user_map[user_id]])
        users.append(user_map[user_id])
        user_subgroups.append(user_subgroup)

    #get ground truth

    print("-----evaluataion--------")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col,sort=False).agg(list).reset_index()
    # ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth
    print(user_pos_item)

    courses = []
    for data in user_pos_item[item_col]:
        courses.append(data)
        
        print(user_subgroups)
        exit(0)
        preds = []
        for user in user_subgroups:
            pred = []
            for subgroups in user:
                for subgroup in subgroups.split(','):
                    pred.append(subgroup)
            preds.append(pred)
        print(preds)
        exit(0)
        group_name = pd.read_csv("subgroups.csv")
        courses = pd.read_csv("courses.csv")
        for cids in courses:
            for cid in cids.split(" "):
                courses[cid]

        out = mapk(labels, preds)
        print("mapk score:" + str(out))

    print("-------------testing--------")
    submit = {"user_id":[],"course_id":[]}
    for user,pred in zip(users,preds):
        submit["user_id"].append(user)
        pred = " ".join(pred)
        submit["course_id"].append(pred)

    submit = pd.DataFrame(submit)
    submit.to_csv(args.save_dir + "result.csv", index=False)

user_embedding = torch.load("./output/user_embedding.pth")
item_embedding = torch.load("./output/user_embedding.pth")
raw_id_maps_path = "./input/hahow/data/raw_id_maps.npy"
match_evaluation(user_embedding, item_embedding, x_eval, all_item, raw_id_maps=raw_id_maps_path, topk=5)
