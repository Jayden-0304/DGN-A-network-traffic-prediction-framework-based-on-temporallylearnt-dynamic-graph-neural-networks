import os
import tensorflow as tf
import numpy as np
import glob

from spektral.datasets import citation
from spektral.layers import GCNConv
import mymodel_dgn as mymodel
from tqdm import tqdm
import time


hparams = {
    'GCNConv_dim': 10,# 10
    'GRU_dim': 100,
    'input_dim': 22,
    'num_nodes': 22,  # 12
    'learning_rate': 0.001,
    'readout_unit': 100,
    'dropout_rate': 0.2
}
topo = "geant_my"
model_name = ("DGN")
batch_size = 32


def _forward_pass(x,model):
    return model(x, training=True)

def train(batch,model):
    with tf.GradientTape() as tape:
        preds = []
        target = []
        for examp in batch:
            x = examp[0]
            y = examp[1]
            prediction = _forward_pass(x,model)
            preds.append(tf.reshape(prediction, [-1]))
            target.append(tf.reshape(y, [-1]))
        preds = tf.reshape(preds, [-1])
        target = tf.reshape(target, [-1])
        loss = tf.keras.losses.MSE(preds, target)
        # regularization_loss = sum(model.losses)
        # print(regularization_loss)
        # loss = loss + regularization_loss
    grad = tape.gradient(loss, model.variables)
    # gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
    gradients = grad
    model.optimizer.apply_gradients(zip(gradients, model.variables))
    return loss

def trainning(times):

    model = mymodel.myModel(hparams)
    model.build()

    model.summary()

    directory_train = "tfrecords_"+topo+"_new/train/"
    directory_validation = "tfrecords_"+topo+"_new/validation/"
    tfrecords = glob.glob(directory_train + '*.tfrecords')
    tfrecords_validation = glob.glob(directory_validation + '*.tfrecords')

    dic = {
        'traffic': tf.io.VarLenFeature(dtype=tf.float32),
        'traffic_shape': tf.io.FixedLenFeature(shape=(2,), dtype=tf.int64),
        'adjacency': tf.io.VarLenFeature(dtype=tf.float32)}

    if not os.path.exists("./Logs_"+topo+"_new_2"):
        os.makedirs("./Logs_"+topo+"_new_2")
    checkpoint_dir = "2_"+topo+"_new_2/models_"+model_name+"_"+str(times)+"/"
    fileLogs = open("./Logs_"+topo+"_new_2/"+topo+"_"+model_name+"_"+str(times)+".txt", "a")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

    min_loss = 1000
    min_iter = 0
    for i in tqdm(range(100)): # 200
        off_set = 0
        while off_set < len(tfrecords):
            if off_set + batch_size < len(tfrecords):
                tfsamples = tfrecords[off_set:off_set + batch_size]
            else:
                tfsamples = tfrecords[off_set:]
            off_set += 32
            batch = []
            for tfr in tfsamples:
                data = tf.data.TFRecordDataset(tfr)
                input = []
                for d in data:
                    x = tf.io.parse_single_example(d, dic)
                    traffic = tf.sparse.to_dense(x['traffic'])  / 7946.
                    traffic = tf.reshape(traffic, x['traffic_shape'])
                    input.append(traffic)
                    a = tf.sparse.to_dense(x['adjacency'])
                    a = tf.reshape(a, x['traffic_shape'])
                a = np.array(a)
                a = np.where(a, a, a.T).astype('f4')
                batch.append([[input[0:-1], a], input[-1]])
            loss = train(batch, model)
        losses = []
        for file in tfrecords_validation:
            data = tf.data.TFRecordDataset(file)
            input = []
            for d in data:
                x = tf.io.parse_single_example(d, dic)
                traffic = tf.sparse.to_dense(x['traffic'])  / 7946.
                traffic = tf.reshape(traffic, x['traffic_shape'])
                input.append(traffic)
                a = tf.sparse.to_dense(x['adjacency'])
                a = tf.reshape(a, x['traffic_shape'])
            a = np.array(a)
            a = np.where(a, a, a.T)
            pred = model([input[0:-1], a])
            target = input[-1]
            pred = tf.reshape(pred, [-1])
            target = tf.reshape(target, [-1])
            loss = tf.keras.losses.MSE(pred, target)
            losses.append(loss)
        mean_loss = np.mean(losses)

        if mean_loss < min_loss:
            min_loss = mean_loss
            min_iter = i

        fileLogs.write(">," + str(mean_loss) + ",\n")
        fileLogs.write("-," + str(i) + ",\n")
        # Store trained model
        checkpoint.save(checkpoint_prefix)
        fileLogs.write("MIN LOSS: " + str(min_loss) + " MODEL_ID: " + str(min_iter) + ",\n")
        fileLogs.flush()

        print('Epoch {}, Loss {:.6f}'.format(i, mean_loss))

        if i - min_iter > 5 and i >= 50:
            break

    checkpoint.save(checkpoint_prefix)

for i in range(10):
    checkpoint_dir = "2_"+topo+"_new_2/models_"+model_name+"_" + str(i+1) + "/"
    if not os.path.exists(checkpoint_dir):
        trainning(i+1)
        break
