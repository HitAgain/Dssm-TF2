# -*- coding: utf-8 -*-
# tf 训练

import datetime
import logging
import os
import time

import tensorflow as tf

from model import DSSMmodel
from preprocess import train_dataset, test_dataset

# 评估函数模型前向计算样本对cos相似度和真实label计算auc
def evaluate(test_data, model):
    test_auc = tf.keras.metrics.AUC()
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64), # query
        tf.TensorSpec(shape=(None, None), dtype=tf.int64), # doc
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64), # label
    ]
    @tf.function(input_signature=train_step_signature)
    def evaluate_step(query, doc, label):
        predictions = model.cosine_sim_predict((query, doc))
        test_auc(label, predictions)
        return predictions
    for (batch, inputs) in enumerate(test_data):
        predictions = evaluate_step(*inputs)
    return test_auc.result().numpy()

def train(train_file_name, test_file_name):
    input_files = [train_file_name]
    total_days = 0
    train_data = train_dataset(
        input_files,
        shuffle=True,
        shuffle_buffer_size=1000000,
        batch_size = 32,
        prefetch_buffer_size=128,
    )
    model = DSSMmodel("./pretrain/c2v")
    learing_rate = tf.keras.experimental.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=2000,
        alpha=0.7,
    )
    optimizer = tf.keras.optimizers.Adam(learing_rate)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    model.compile(loss="categorical_crossentropy", optimizer = optimizer, metrics=["accuracy"])
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(
        query,
        doc_pos,
        doc_neg,
        label
    ):
        with tf.GradientTape() as tape:
            predictions = model((query, doc_pos, doc_neg))

            loss = tf.keras.losses.MSE(label, predictions)
            # loss = tf.keras.losses.binary_crossentropy(label, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_auc(label, predictions)

    # train
    train_loss.reset_states()
    train_auc.reset_states()
    all_batch_per_epoch = 100
    epoch = 0
    best_auc = 0.5
    for (batch, inputs) in enumerate(train_data):
        if batch % all_batch_per_epoch == 0:
            epoch += 1
            logging.info("start training epoch {}".format(epoch))
        train_step(*inputs)
        # evaluate
        if batch % 20 == 0:
            test_input_files = [test_file_name]
            test_data = test_dataset(
                test_input_files,
                batch_size = 32,
                prefetch_buffer_size=256,
            )
            test_auc = evaluate(
                test_data,
                model
            )
            logging.info("batch:{}, test_auc:{}".format(batch, test_auc))
            if test_auc > best_auc:
                tf.saved_model.save(model,
                                    "./output",
                                    signatures={
                                        'query_emb': model.query,
                                        'doc_emb': model.doc
                                    })
                best_auc = test_auc
    logging.info("DSSM Model Best Auc {}".format(best_auc))

if __name__ == "__main__":
    logging.info("start train")
    train("./train/train.txt")
    logging.info("end train")
