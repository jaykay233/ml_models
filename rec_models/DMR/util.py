# coding:utf8
import paddle.nn as nn
import paddle
import numpy as np

# user feature size
user_size = 1141730
cms_segid_size = 97
cms_group_id_size = 13
final_gender_code_size = 3
age_level_size = 7
pvalue_level_size = 4
shopping_level_size = 4
occupation_size = 3
new_user_class_level_size = 5

# item feature size
adgroup_id_size = 846812
cate_size = 12978
campaign_id_size = 423437
customer_size = 255876
brand_size = 461529

# context feature size
btag_size = 5
pid_size = 2

# embedding_size
main_embedding_size = 32
other_embedding_size = 8

def create_embedding(l):
    assert len(l) == 2
    H = l[0]
    E = l[1]
    embedding = paddle.nn.Embedding(H, E, sparse=True)
    w0 = np.random.normal(0, 1, (H, E)).astype(np.float32)
    embedding.weight.set_value(w0)
    return embedding


def create_tensor(x, stop_gradient=True):
    return paddle.to_tensor(x, stop_gradient=stop_gradient)


def expand_dims(data, dims):
    dim_list = data.shape
    if dims<0:
        dims += len(dim_list) + 1
    new_dims = dim_list[:dims] + [1] + (dim_list[dims:] if dims < len(dim_list) else [])
    # print(new_dims)
    return paddle.reshape(data, new_dims)
