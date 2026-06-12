
import nttg
import tensorflow as tf
from spektral.layers import GCNConv
from spektral.layers import GATConv
from keras import regularizers


class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams
        self.GCN = []
        for i in range(1*2):
            self.GCN.append(GCNConv(self.hparams['GCNConv_dim'],
                                    activation=tf.nn.selu))
        # A1表示学习邻接矩阵
        self.A1 = nttg.nttg(self.hparams['input_dim'],activation=tf.nn.relu6, name="FirstLayer")
        self.RNN1 = tf.keras.layers.GRUCell(self.hparams['GRU_dim'], dtype=tf.float32)
        self.RNN2 = tf.keras.layers.GRUCell(self.hparams['GRU_dim'], dtype=tf.float32)

        self.Readout = tf.keras.layers.Dense(self.hparams['input_dim']*self.hparams['num_nodes'],activation=tf.nn.selu, kernel_regularizer=regularizers.l2(0.1),
                                                                                name="Readout2")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])



    def build(self, input_shape=None):
        self.A1.build(input_shape=tf.TensorShape([None, self.hparams['input_dim']]))

        for i in range(1*2):
            if(i<2):
                self.GCN[i].build(input_shape=[tf.TensorShape([None, self.hparams['input_dim']]),
                                               tf.TensorShape([None, self.hparams['num_nodes']])])
            else:
                self.GCN[i].build(input_shape=[tf.TensorShape([None, self.hparams['GCNConv_dim']]),
                                               tf.TensorShape([None, self.hparams['num_nodes']])])
        self.RNN1.build(input_shape=tf.TensorShape([None,self.hparams['GCNConv_dim']*self.hparams['input_dim']]))
        self.RNN2.build(input_shape=tf.TensorShape([None, self.hparams['GRU_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['GRU_dim']])
        self.built = True

    # @tf.function
    def call(self, inputs, training=False):
        # Define the forward pass
        input_sequence = inputs[0]
        A = inputs[1]
        A = self.preprocess(A)[0]
        out = []
        ht1 = []
        ht2 = []
        for j in range(len(input_sequence)):
            out1 = input_sequence[j]
            out2 = input_sequence[j]
            gcn_layer = GCNConv(channels=22)
            output = gcn_layer([input_sequence[j], A])
            a = self.A1(output)
            a1 = self.preprocess(a)[0]
            a2 = tf.transpose(a)
            a2 = self.preprocess(a2)[0]
            for k in range(1):
                out1 = self.GCN[k*2]([out1,a1])
                out2 = self.GCN[k*2+1]([out2,a2])
            out1 = out1+out2

            if j == 0:
                input2 = tf.reshape(out1, [1, self.hparams['num_nodes'] * self.hparams['GCNConv_dim']])
                h = tf.zeros([1, self.hparams['GRU_dim']])
                out2, h1 = self.RNN1(input2, [h])
                ht1.append(h1[0])
                out3, h2 = self.RNN2(out2, [h])
                ht2.append(h2[0])
            else:
                input2 = tf.reshape(out1, [1, self.hparams['num_nodes'] * self.hparams['GCNConv_dim']])
                out2, h1 = self.RNN1(input2, [ht1[-1]])
                ht1.append(h1[0])
                out3, h2 = self.RNN2(out2, [ht2[-1]])
                ht2.append(h2[0])
                if j == (len(input_sequence) - 1):
                    out4 = self.Readout(out3,training=training)
                    out = out4
        out = tf.reshape(out, [1, self.hparams['num_nodes'], self.hparams['input_dim']])

        return out

    def preprocess(self,A):
        out = A
        out = out + tf.linalg.diag(tf.ones((1, A.shape[0])))
        s = tf.reduce_sum(out, 1)
        out = out / s
        return out
