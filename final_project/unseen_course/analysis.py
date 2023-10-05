import numpy as np
import torch

raw_id_maps_path = "input/hahow/data/" + "raw_id_maps.npy"
user_map, item_map = np.load(raw_id_maps_path, allow_pickle=True)
print(user_map[1293])

raw_id_maps_path = "./output/" + "user_embedding.pth"
user_embedding = torch.load(raw_id_maps_path)

print(user_embedding.size())
print(user_embedding)



raw_id_maps_path = "./output/" + "item_embedding.pth"
item_embedding = torch.load(raw_id_maps_path)

print(item_embedding.size())
print(item_embedding)