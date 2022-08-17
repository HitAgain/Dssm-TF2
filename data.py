# -*- coding: utf-8 -*-
# tf-data数据


import tensorflow as tf

@tf.function
def map_fn_train(line):
    data = tf.strings.split(line, "\t")
    query_token_ids = tf.strings.to_number(tf.strings.split(data[0], "|"), out_type=tf.int64)
    doc_pos_token_ids = tf.strings.to_number(tf.strings.split(data[1], "|"), out_type=tf.int64)
    doc_neg_token_ids = tf.strings.to_number(tf.strings.split(data[2], "|"), out_type=tf.int64)
    label = tf.strings.to_number(tf.strings.split(data[3], "|"), out_type=tf.int64)
    return (
        query_token_ids,
        doc_pos_token_ids,
        doc_neg_token_ids,
        label,
    )

@tf.function
def map_fn_test(line):
    data = tf.strings.split(line, "\t")
    query_token_ids = tf.strings.to_number(tf.strings.split(data[0], "|"), out_type=tf.int64)
    doc_token_ids = tf.strings.to_number(tf.strings.split(data[1], "|"), out_type=tf.int64)
    label = tf.strings.to_number([data[2]], out_type=tf.int64)
    return (
        query_token_ids,
        doc_token_ids,
        label
    )

def train_dataset(
    input_files,
    shuffle=True,
    shuffle_buffer_size=100,
    batch_size=64,
    prefetch_buffer_size=128,
    num_parallel_reads=8,
):
    # 文件路径或者文件路径列表的Dataset形式
    dataset = tf.data.Dataset.list_files(input_files)
    # epoch = 4
    dataset = dataset.shuffle(shuffle_buffer_size,
                              reshuffle_each_iteration=True).repeat(4)
    # 并行按行读入数据
    lines_dataset = tf.data.TextLineDataset(
        dataset,
        num_parallel_reads=num_parallel_reads,
    ).map(map_fn_train,
          num_parallel_calls=num_parallel_reads)  # ((batch_size, seq_len), (batch_size, 1))

    if shuffle:
        lines_dataset = lines_dataset.shuffle(shuffle_buffer_size,
                                              seed=700,
                                              reshuffle_each_iteration=True)

    lines_dataset = lines_dataset
        .padded_batch(
            batch_size,
            padded_shapes = -1,
            padding_values = 0,
            drop_remainder = False,
            name=None
        )
        .prefetch(buffer_size=prefetch_buffer_size)
    return lines_dataset


def test_dataset(
    input_files,
    batch_size=64,
    prefetch_buffer_size=128,
    num_parallel_reads=8,
):
    dataset = tf.data.Dataset.list_files(input_files)
    lines_dataset = tf.data.TextLineDataset(
        dataset,
        num_parallel_reads=num_parallel_reads,
    )
    lines_dataset = lines_dataset
        .map(map_fn_test, num_parallel_calls=num_parallel_reads)
        .padded_batch(
            batch_size,
            padded_shapes = -1,
            padding_values = 0,
            drop_remainder = True,
            name=None)
        .prefetch(
            buffer_size=prefetch_buffer_size)
    return lines_dataset
