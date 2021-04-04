# coding:utf8
import paddle.nn as nn
import paddle
import numpy as np
from util import *


def parse_feature(feature):
    ### history
    btag_his = paddle.cast(feature[:, 0:50], 'int32')
    cate_his = paddle.cast(feature[:, 50:100], 'int32')
    brand_his = paddle.cast(feature[:, 100:150], 'int32')
    mask = paddle.cast(feature[:, 150:200], 'int32')
    match_mask = paddle.cast(feature[:, 200:250], 'int32')

    ## user
    uid = paddle.cast(feature[:, 250], 'int32')
    cms_segid = paddle.cast(feature[:, 251], 'int32')
    cms_group_id = paddle.cast(feature[:, 252], 'int32')
    final_gender_code = paddle.cast(feature[:, 253], 'int32')
    age_level = paddle.cast(feature[:, 254], 'int32')
    pvalue_level = paddle.cast(feature[:, 255], 'int32')
    shopping_level = paddle.cast(feature[:, 256], 'int32')
    occupation = paddle.cast(feature[:, 257], 'int32')
    new_user_class_level = paddle.cast(feature[:, 258], 'int32')

    ## item
    mid = paddle.cast(feature[:, 259], 'int32')
    cate_id = paddle.cast(feature[:, 260], 'int32')
    campaign_id = paddle.cast(feature[:, 261], 'int32')
    customer = paddle.cast(feature[:, 262], 'int32')
    brand = paddle.cast(feature[:, 263], 'int32')
    price = expand_dims(paddle.cast(feature[:, 264], 'float32'), 1)

    pid = paddle.cast(feature[:, 265], 'int32')
    return btag_his, cate_his, brand_his, mask, match_mask, uid, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level, mid, cate_id, campaign_id, customer, brand, price, pid


class Model_DMR(nn.Layer):
    def __init__(self):
        super().__init__()
        self.uid_embeddings_var = create_embedding([user_size, main_embedding_size])
        self.mid_embeddings_var = create_embedding([adgroup_id_size, main_embedding_size])
        self.cat_embeddings_var = create_embedding([cate_size, main_embedding_size])
        self.brand_embeddings_var = create_embedding([brand_size, main_embedding_size])
        self.btag_embeddings_var = create_embedding([btag_size, other_embedding_size])
        self.dm_tag_embeddings_var = create_embedding([btag_size, other_embedding_size])
        self.campaign_id_embeddings_var = create_embedding([campaign_id_size, main_embedding_size])
        self.customer_embeddings_var = create_embedding([customer_size, main_embedding_size])
        self.cms_segid_embeddings_var = create_embedding([cms_segid_size, other_embedding_size])
        self.cms_group_id_embeddings_var = create_embedding([cms_group_id_size, other_embedding_size])
        self.final_gender_code_embeddings_var = create_embedding([final_gender_code_size, other_embedding_size])
        self.age_level_embeddings_var = create_embedding([age_level_size, other_embedding_size])
        self.pvalue_level_embeddings_var = create_embedding([pvalue_level_size, other_embedding_size])
        self.shopping_level_embeddings_var = create_embedding([shopping_level_size, other_embedding_size])
        self.occupation_embeddings_var = create_embedding([occupation_size, other_embedding_size])
        self.new_user_class_level_embeddings_var = create_embedding([new_user_class_level_size, other_embedding_size])
        self.pid_embddings_var = create_embedding([pid_size, other_embedding_size])

        self.position_his = create_tensor(np.arange(50),stop_gradient=True)
        self.position_embeddings_var = create_embedding([50, other_embedding_size])
        self.position_his_eb = self.position_embeddings_var(self.position_his)

        # print("pos -1: ",self.position_his_eb.shape)
        self.dm_position_his = create_tensor(np.arange(50),stop_gradient=True)
        self.dm_position_embeddings_var = create_embedding([50, other_embedding_size])
        self.dm_position_his_eb = self.dm_position_embeddings_var(self.dm_position_his)

        ### user2item
        self.dm_item_vectors = create_embedding([cate_size, main_embedding_size])
        self.dm_item_biases = create_tensor(np.zeros(cate_size), stop_gradient=True)


    def deep_match(self, item_his_eb, context_his_eb, mask, match_mask, mid_his_batch, EMBEDDING_DIM, item_vectors,
                   item_biases):
        """
        user2item
        :param item_his_eb:
        :param context_his_eb:
        :param mask:
        :param match_mask:
        :param mid_his_batch:
        :param EMBEDDING_DIM:
        :param item_vectors:
        :param item_biases:
        :return:
        """
        query = context_his_eb
        query = paddle.fluid.layers.fc(query,main_embedding_size * 2, num_flatten_dims=2)
        query = paddle.fluid.layers.prelu(query,mode='channel')

        # print(query.shape)
        # print(item_his_eb.shape)
        inputs = paddle.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)  ## B * T * E
        att1 = paddle.fluid.layers.fc(inputs, 80,num_flatten_dims=2, act='sigmoid')
        att2 = paddle.fluid.layers.fc(att1, 40, num_flatten_dims=2,act='sigmoid')
        att3 = paddle.fluid.layers.fc(att2, 1,num_flatten_dims=2)

        scores = paddle.transpose(att3, [0, 2, 1])

        ## mask
        bool_mask = paddle.equal(mask, paddle.ones_like(mask))
        key_masks = expand_dims(bool_mask, 1)
        paddings = paddle.ones_like(scores) * (-2 ** 32 + 1)
        scores = paddle.where(key_masks, scores, paddings)

        ## tril
        scores_tile = paddle.tile(paddle.sum(scores, axis=1), [1, scores.shape[-1]])
        scores_tile = paddle.reshape(scores_tile, [-1, scores.shape[-1], scores.shape[-1]])
        diag_vals = paddle.ones_like(scores_tile)
        tril = paddle.tril(diag_vals)
        paddings = paddle.ones_like(tril) * (-2 ** 32 + 1)
        scores_tile = paddle.where(paddle.equal(tril, paddle.zeros_like(tril)), paddings, scores_tile)
        scores_tile = paddle.nn.functional.softmax(scores_tile)
        att_dim_item_his_eb = paddle.matmul(scores_tile, item_his_eb)

        dnn_layer1 = paddle.fluid.layers.fc(att_dim_item_his_eb, EMBEDDING_DIM,num_flatten_dims=2)
        dnn_layer1 = paddle.fluid.layers.prelu(dnn_layer1,mode='channel')

        user_vector = dnn_layer1[:, -1, :]
        user_vector2 = dnn_layer1[:, -2, :] * paddle.reshape(match_mask, [-1, match_mask.shape[1], 1])[:, -2, :]
        num_sampled = 2000
        # print("user_vector: ",user_vector.shape)
        # print("user_vector2: ",user_vector2.shape)
        # print("item_vector: ",item_vectors.weight.shape)
        # print("item_biases: ",item_biases.shape)
        logits = paddle.matmul(user_vector2, item_vectors.weight, transpose_y=True) + item_biases


        loss = paddle.mean(paddle.fluid.layers.sampled_softmax_with_cross_entropy(
            label=paddle.cast(paddle.reshape(mid_his_batch[:, -1], [-1, 1]), 'int64'),
            logits=logits, num_samples=num_sampled))
        return loss, user_vector, scores

    def dmr_fcn_attention(self, item_eb, item_his_eb, context_his_eb, mask, mode='SUM'):
        """
        item2item
        :param item_eb:
        :param item_his_eb:
        :param context_his_eb:
        :param mask:
        :param mode:
        :return:
        """
        mask = paddle.equal(mask, paddle.ones_like(mask))
        item_eb_tile = paddle.tile(item_eb, [1, mask.shape[1]])
        item_eb_tile = paddle.reshape(item_eb_tile, [-1, mask.shape[1], item_eb.shape[-1]])  # B, T, E
        if context_his_eb is None:
            query = item_eb_tile
        else:
            query = paddle.concat([item_eb_tile, context_his_eb], axis=-1)
        query = paddle.fluid.layers.fc(query, item_his_eb.shape[-1],num_flatten_dims=2)
        query = paddle.fluid.layers.prelu(query,mode='channel')
        dmr_all = paddle.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)
        att_layer_1 = paddle.fluid.layers.fc(dmr_all, 80, num_flatten_dims=2, act='sigmoid')
        att_layer_2 = paddle.fluid.layers.fc(att_layer_1, 40, num_flatten_dims=2, act='sigmoid')
        att_layer_3 = paddle.fluid.layers.fc(att_layer_2, 1, num_flatten_dims=2, act=None)
        att_layer_3 = paddle.reshape(att_layer_3, [-1, 1, item_his_eb.shape[1]])
        scores = att_layer_3


        ##Mask
        key_masks = expand_dims(mask, 1)
        paddings = paddle.ones_like(scores) * (-2 ** 32 + 1)
        paddings_no_softmax = paddle.zeros_like(scores)
        scores = paddle.where(key_masks, scores, paddings)
        scores_no_softmax = paddle.where(key_masks, scores, paddings_no_softmax)

        scores = paddle.fluid.layers.softmax(scores)

        if mode == 'SUM':
            output = paddle.matmul(scores, item_his_eb)
            output = paddle.sum(output, axis=1)
        else:
            scores = paddle.reshape(scores, [-1, item_his_eb.shape[1]])
            output = item_his_eb * expand_dims(scores, -1)
            output = paddle.reshape(output, item_his_eb.shape)
        return output, scores, scores_no_softmax

    def build_fcn_net(self,inp):
        """
        build fcn_net
        :param inp:
        :return:
        """
        inp = paddle.fluid.layers.batch_norm(input=inp)
        dnn0 = paddle.fluid.layers.fc(inp,512,act=None)
        dnn0 = paddle.fluid.layers.prelu(dnn0,mode='channel')
        dnn1 = paddle.fluid.layers.fc(dnn0,256,act=None)
        dnn1 = paddle.fluid.layers.prelu(dnn1,mode='channel')
        dnn2 = paddle.fluid.layers.fc(dnn1,128)
        dnn2 = paddle.fluid.layers.prelu(dnn2,mode='channel')
        dnn3 = paddle.fluid.layers.fc(dnn2,1,act=None)
        return paddle.fluid.layers.sigmoid(dnn3)

    def forward(self, feature):
        # print(feature)
        btag_his, cate_his, brand_his, mask, match_mask, uid, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level, mid, cate_id, campaign_id, customer, brand, price, pid = parse_feature(
            feature)
        uid_batch_embedded = self.uid_embeddings_var(uid)

        mid_batch_embedded = self.mid_embeddings_var(mid)

        cat_batch_embedded = self.cat_embeddings_var(cate_id)
        cat_his_batch_embedded = self.cat_embeddings_var(cate_his)

        brand_batch_embedded = self.brand_embeddings_var(brand)
        brand_his_batch_embedded = self.brand_embeddings_var(brand_his)

        btag_his_batch_embedded = self.btag_embeddings_var(btag_his)
        dm_btag_his_batch_embedded = self.dm_tag_embeddings_var(btag_his)


        campaign_id_batch_embedded = self.campaign_id_embeddings_var(campaign_id)

        customer_batch_embedded = self.customer_embeddings_var(customer)

        cms_segid_batch_embedded = self.cms_segid_embeddings_var(cms_segid)

        cms_group_id_batch_embedded = self.cms_group_id_embeddings_var(cms_group_id)

        final_gender_code_batch_embedded = self.final_gender_code_embeddings_var(final_gender_code)

        age_level_batch_embedded = self.age_level_embeddings_var(age_level)

        pvalue_level_batch_embedded = self.pvalue_level_embeddings_var(pvalue_level)

        shopping_level_batch_embedded = self.shopping_level_embeddings_var(shopping_level)

        occupation_batch_embedded = self.occupation_embeddings_var(occupation)

        new_user_class_level_batch_embedded = self.new_user_class_level_embeddings_var(new_user_class_level)

        pid_batch_embedded = self.pid_embddings_var(pid)

        user_feat = paddle.concat([uid_batch_embedded, cms_segid_batch_embedded, cms_group_id_batch_embedded,
                                   final_gender_code_batch_embedded, age_level_batch_embedded,
                                   pvalue_level_batch_embedded, shopping_level_batch_embedded,
                                   occupation_batch_embedded, new_user_class_level_batch_embedded], axis=-1)
        item_his_emb = paddle.concat([cat_his_batch_embedded, brand_his_batch_embedded], axis=-1)
        item_his_emb_sum = paddle.sum(item_his_emb, axis=1)

        # print(mid_batch_embedded.shape, cat_batch_embedded.shape, brand_batch_embedded.shape, campaign_id_batch_embedded.shape, customer_batch_embedded.shape,price.shape)

        item_feat = paddle.concat(
            [mid_batch_embedded, cat_batch_embedded, brand_batch_embedded, campaign_id_batch_embedded,
             customer_batch_embedded, price], axis=-1)
        item_eb = paddle.concat([cat_batch_embedded, brand_batch_embedded], axis=-1)
        context_feat = pid_batch_embedded
        # print("pos0:",self.position_his_eb.shape)
        position_his_eb = paddle.tile(self.position_his_eb, [mid.shape[0], 1])
        # print("pos1: ",position_his_eb.shape)
        position_his_eb = paddle.reshape(position_his_eb, [mid.shape[0], -1, position_his_eb.shape[1]])
        # print("pos2: ",position_his_eb.shape)
        dm_position_his_eb = paddle.tile(self.dm_position_his_eb, [mid.shape[0], 1])
        dm_position_his_eb = paddle.reshape(dm_position_his_eb,
                                                 [mid.shape[0], -1, dm_position_his_eb.shape[1]])

        position_his_eb = paddle.concat([position_his_eb, btag_his_batch_embedded], axis=-1)
        dm_position_his_eb = paddle.concat([dm_position_his_eb, dm_btag_his_batch_embedded], axis=-1)

        # print("position_his_eb: ",position_his_eb.shape)
        # print("btag_his_batch_embedded: ",btag_his_batch_embedded.shape)



        # print(item_his_emb.shape)
        # print(self.dm_position_his_eb.shape)

        aux_loss, dm_user_vector, scores = self.deep_match(item_his_emb, dm_position_his_eb, mask,
                                                           paddle.cast(match_mask, 'float32'), cate_his,
                                                           main_embedding_size, self.dm_item_vectors,
                                                           self.dm_item_biases)
        aux_loss *= 0.1
        dm_item_vec = self.dm_item_vectors(cate_id)
        rel_u2i = paddle.sum(dm_user_vector * dm_item_vec, axis=-1, keepdim=True)

        ## item2item
        att_outputs, alphas, scores_unnorm = self.dmr_fcn_attention(item_eb, item_his_emb, position_his_eb, mask)
        rel_i2i = expand_dims(paddle.sum(scores_unnorm,[1,2]),-1)
        scores = paddle.sum(alphas,axis=1)

        # print(user_feat.shape,item_feat.shape,context_feat.shape,item_his_emb_sum.shape,item_eb.shape,rel_u2i.shape,rel_i2i.shape,att_outputs.shape)
        inp = paddle.concat([user_feat,item_feat,context_feat,item_his_emb_sum,item_eb*item_his_emb_sum,rel_u2i,rel_i2i,att_outputs],-1)

        return self.build_fcn_net(inp)