# -*- coding: utf-8 -*-
# tf模型

import tensorflow as tf
from gensim.models import KeyedVectors

# model build
class DSSMmodel(tf.keras.Model):
    def __init__(
        self,
        pretrain_char2vec_file,
        rate=0.1,
    ):
        super(DSSMmodel, self).__init__()
        self.char2vec_weights = load_char2vec(pretrain_char2vec_file)
        self.char_embedding = tf.keras.layers.Embedding(np.shape(self.char2vec_weights)[0],
                                                       np.shape(self.char2vec_weights)[1],
                                                       weights=[self.char2vec_weights],
                                                       trainable=self.char_weights_trainable,
                                                       mask_zero = True,
                                                       name = "char_embedding_layer")

        self.left_forward = tf.keras.Sequential([
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(128, activation="sigmoid"),
        ])

        self.right_forward = tf.keras.Sequential([
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(128, activation="sigmoid"),
        ])


    @staticmethod
    # 加载预训练字向量
    def load_char2vec(char2vec_file_path):
        model = KeyedVectors.load_word2vec_format(char2vec_file_path, binary=False)
        weights = model.syn0
        return weights

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, ARGS.max_query_seq_len), dtype=tf.int64)
    # ])
    def cbow_emb(self, x):
        # [batch_size, input_seq_len]
        mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)
        # [batch_size, input_seq_len, 1]
        mask_b = tf.expand_dims(mask, axis=-1)
        # [batch_size, input_seq_len, hidden_size]
        x_emb = self.char_embedding(x)
        # [batch_size, hidden_size]
        return tf.reduce_sum(tf.math.multiply(x_emb, mask_b), axis=1)

    # query塔输出l2规范化向量
    def query_tower(self, query_ids, training):
        query_box_emb = self.cbow_emb(query_ids)
        left_output = self.left_forward(query_box_emb, training=training)
        left_norm = tf.nn.l2_normalize(left_output, axis=1, name='left_norm')
        return left_norm

    # doc塔输出l2规范化向量
    def doc_tower(self, doc_ids, training):
        doc_emb = self.cbow_emb(doc_ids)
        right_output = self.right_forward(doc_emb, training=training)
        rigth_norm = tf.nn.l2_normalize(right_output, axis=1, name='right_norm')
        return rigth_norm

    # train output
    def call(self, inputs, training = True):
        query, doc_pos, doc_neg = inputs
        query_norm = self.query_tower(query, training)
        doc_norm_pos = self.doc_tower(doc_pos, training)
        doc_norm_neg = self.doc_tower(doc_neg, training)
        query_pos_doc_cosine = tf.keras.layers.dot([query_norm, doc_norm_pos], axes=1, normalize=False)
        query_neg_doc_cosine = tf.keras.layers.dot([query_norm, doc_norm_neg], axes=1, normalize=False)
        concat_cosine = tf.keras.layers.concatenate([query_pos_doc_cosine, query_neg_doc_cosine])
        concat_cosine = tf.keras.layers.Reshape((2, 1))(concat_cosine)
        # gamma系数
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = tf.keras.layers.Conv1D(1, 1, padding="same", input_shape=(2, 1), activation="linear", use_bias=False, weights=[weight])(
            concat_cosine)
        # [batch_size, 2]
        with_gamma = tf.keras.layers.Reshape((2, ))(with_gamma)
        # softmax
        sim_prob = tf.keras.layers.Activation("softmax")(with_gamma)
        return sim_prob

    # evaluate output
    def cosine_sim_predict(self, inputs, training = False):
        query, doc = inputs
        query_norm = self.query_tower(query, training)
        doc_norm = self.doc_tower(doc, training)
        #cosine_sim = tf.keras.layers.dot([query_norm, doc_norm_pos], axes=1, normalize=False)
        return tf.reduce_sum(tf.multiply(query_norm, doc_norm),
                             axis=1,
                             keepdims=True,
                             name='cosine_sim')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='query')
    ])
    def query(self, query, training = False):
        return {'query_serving': self.query_tower(query, training)}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='doc')
    ])
    def doc(self, doc, training = False):
        return {'doc_serving': self.doc_tower(doc, training)}
