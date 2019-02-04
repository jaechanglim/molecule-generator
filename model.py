import numpy as np
import tensorflow as tf
import numpy as np
import threading

class CVAE():
    def __init__(self,
                 vocab_size,
                 args
                  ):

        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        
        self._create_network()


    def _create_network(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size, None])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, None])
        self.L = tf.placeholder(tf.int32, [self.batch_size])
        

        encoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        
        with tf.variable_scope('rnn'):
            encode_cell=[]
            for i in encoded_rnn_size[:]:
                encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)
        
        self.weights = {}
        self.biases = {}


        self.weights['softmax'] = tf.get_variable("softmaxw", initializer=tf.random_uniform(shape=[encoded_rnn_size[-1], self.vocab_size], minval = -0.1, maxval = 0.1))       
        
        self.biases['softmax'] =  tf.get_variable("softmaxb", initializer=tf.zeros(shape=[self.vocab_size]))

        self.embedding_encode = tf.get_variable(name = 'encode_embedding', shape = [self.unit_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        
        self.decoded, decoded_logits = self.rnn()

        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)
        self.reconstr_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=decoded_logits, targets=self.Y, weights=weights))
        
        # Loss
        self.loss = self.reconstr_loss 
        #self.loss = self.reconstr_loss 
        optimizer    = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.loss)
        
        self.mol_pred = tf.argmax(self.decoded, axis=2)
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=None)
        #tf.train.start_queue_runners(sess=self.sess)
        print ("Network Ready")

    def rnn(self): 
        seq_length=tf.shape(self.X)[1]
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        self.initial_rnn_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((self.batch_size, self.unit_size)), tf.zeros((self.batch_size, self.unit_size))) for i in range(3)])
        Y, self.output_rnn_state = tf.nn.dynamic_rnn(self.encoder, X, dtype=tf.float32, scope = 'rnn', sequence_length = self.L, initial_state=self.initial_rnn_state)
        Y = tf.reshape(Y, [self.batch_size*seq_length, -1])
        Y = tf.matmul(Y, self.weights['softmax'])+self.biases['softmax']
        Y_logits = tf.reshape(Y, [self.batch_size, seq_length, -1])
        Y = tf.nn.softmax(Y_logits)
        return Y, Y_logits

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)
        #print("model saved to '%s'" % (ckpt_path))

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def train(self, x, y, l):
        _, loss = self.sess.run([self.opt, self.loss], feed_dict = {self.X :x, self.Y:y, self.L : l})
        return loss
    
    def test(self, x, y, l):
        mol_pred, loss  = self.sess.run([self.mol_pred, self.loss], feed_dict = {self.X :x, self.Y:y, self.L : l})
        return loss

    def sample(self, start_codon, seq_length):
        l = np.ones((self.batch_size)).astype(np.int32)
        x=start_codon
        preds = []
        for i in range(seq_length):
            if i==0:
                x, state = self.sess.run([self.decoded, self.output_rnn_state], feed_dict = {self.X:x, self.L : l})
            else:
                x, state = self.sess.run([self.decoded, self.output_rnn_state], feed_dict = {self.X:x, self.L : l, self.initial_rnn_state:state})
            sampled_x = []
            for j in range(len(x)):
                prob = x[j,0].tolist()
                norm0 = sum(prob)
                prob = [i/norm0 for i in prob]
                index = np.random.choice(range(np.shape(x)[-1]), 1, p=prob)
                sampled_x.append(index)
            x = np.array(sampled_x)
            #x = np.argmax(x,-1)

            preds.append(x)
        return np.concatenate(preds,1).astype(int).squeeze()
