import cntk as C
import numpy as np
from helpers import *
from cntk.layers import *
from cntk.layers.sequence import * 
from cntk.layers.typing import *
from cntk.debugging import debug_model
import pickle
import importlib
import os

class PolyMath:
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars)
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.w2v_hidden_dim = model_config['w2v_hidden_dim']
        self.convs = model_config['char_convs']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']
        self.use_cudnn = model_config['use_cudnn']
        self.use_sparse = True
        
        # Source and target inputs to the model
        inputAxis = C.Axis('inputAxis')
        outputAxis = C.Axis('outputAxis')
        InputSequence = C.layers.SequenceOver[inputAxis]
        OutputSequence = C.layers.SequenceOver[outputAxis]

        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def embed(self):
        # load glove
        npglove = np.zeros((self.wg_dim, self.w2v_hidden_dim), dtype=np.float32)
        with open(os.path.join(self.abs_path, 'glove.6B.100d.txt'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if word in self.vocab:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(len(self.vocab) - self.wg_dim, self.w2v_hidden_dim), init=C.glorot_uniform(), name='TrainableE')
        
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        return func

    def input_layer(self,cgw,cnw,cc,qgw,qnw,qc):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()

        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = C.splice(
            C.reshape(self.charcnn(input_chars), self.convs),
            self.embed()(input_glove_words, input_nonglove_words), name='splice_embed')
        processed = C.layers.Sequential([For(range(2), lambda: OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn'))])(embedded)
        
        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        
        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})
        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')
        
    def gated_attention_gru_layer(self, context, query):
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        c_processed = C.placeholder(shape=(2*self.hidden_dim,))

        #gate weight
        Wg = C.parameter(shape=(4*self.hidden_dim, 4*self.hidden_dim))
        att_gru = C.layers.GRU(2*self.hidden_dim)
        attention_model = C.layers.AttentionModel(self.hidden_dim, name='attention_model')
        
        @C.Function
        def out_func0(att_input, enc_input):
            enc_input2 = enc_input
            @C.Function
            def gru_with_attentioin(dh, x):
                c_att = attention_model(att_input, x)
                x = C.splice(x, c_att)
                x = C.element_times(x, C.sigmoid(C.times(x, Wg)))
                return att_gru(dh, x)
            att_context = Recurrence(gru_with_attentioin)(enc_input2)
            return att_context
        att_context = out_func0(q_processed, c_processed)
        return C.as_block(
            att_context,
            [(c_processed, context), (q_processed, query)],
            'gated_attention_gru_layer',
            'gated_attention_gru_layer')
            
    def matching_attention_layer(self, attention_context):
        att_context = C.placeholder(shape=(2*self.hidden_dim,))
        #matching layer
        matching_model = C.layers.AttentionModel(attention_dim=self.hidden_dim, name='attention_model')
        #gate weight
        Wg = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim))
        #gru
        att_gru = C.layers.GRU(self.hidden_dim)
        @C.Function
        def out_func1(att_input, enc_input):
            enc_input2 = enc_input
            @C.Function
            def bigru_with_match(dh, x):
                c_att = matching_model(att_input, dh)
                x = C.splice(x, c_att)
                x = C.element_times(x, C.sigmoid(C.times(x, Wg)))
                return att_gru(dh, x)
            return C.splice(C.layers.Recurrence(bigru_with_match)(enc_input2),
                        C.layers.Recurrence(bigru_with_match, go_backwards=True)(enc_input2),
                        name="bigru_with_match")
        match_context = out_func1(att_context, att_context)
        return C.as_block(
            match_context,
            [(att_context, attention_context)],
            'matching_attention_layer',
            'matching_attention_layer')
    
    def output_layer(self, query, match_context):
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        mat_context = C.placeholder(shape=(2*self.hidden_dim,))
        
        #output layer
        r_q = question_pooling(q_processed, 2*self.hidden_dim) #shape n*(2*self.hidden_dim)
        p1_logits = attention_weight(mat_context, r_q, 2*self.hidden_dim)
        attention_pool = C.sequence.reduce_sum(p1_logits * mat_context)
        state = C.layers.GRU(2*self.hidden_dim)(attention_pool, r_q)
        p2_logits = attention_weight(mat_context, state, 2*self.hidden_dim)
        
        @C.Function
        def start_ave_point(p1_logits, p2_logits, point):
            @C.Function
            def start_ave(last, now):
                now = now + last - last
                new_start = now * C.sequence.gather(p2_logits, point)
                point = C.sequence.future_value(point)
                return new_start
            start_logits_ave = C.layers.Recurrence(start_ave)(p1_logits)
            return start_logits_ave
        point = C.sequence.is_first(p1_logits)
        point = C.layers.Sequential([For(range(2), lambda: C.layers.Recurrence(C.plus))])(point)
        point = C.greater(C.constant(16), point)
        start_logits_ave = start_ave_point(p1_logits, p2_logits, point)
        
        @C.Function
        def end_ave_point(p1_logits, p2_logits, point):
            @C.Function
            def end_ave(last, now):
                now = now + last - last
                new_end = now * C.sequence.gather(p2_logits, point)
                point = C.sequence.past_value(point)
                return new_end
            end_logits_ave = C.layers.Recurrence(end_ave, go_backwards=True)(p2_logits)
            return end_logits_ave
        point = C.sequence.is_last(p1_logits)
        point = C.layers.Sequential([For(range(2), lambda: C.layers.Recurrence(C.plus, go_backwards=True))])(point)
        point = C.greater(C.constant(16),point)
        end_logits_ave = end_ave_point(p1_logits, p2_logits, point)
        
        start_logits = seq_hardmax(start_logits_ave)
        end_logits = seq_hardmax(end_logits_ave)
        '''
        start_logits = seq_hardmax(p1_logits)
        end_logits = seq_hardmax(p2_logits)
        '''
        return C.as_block(
            C.combine([start_logits, end_logits]),
            [(q_processed, query), (mat_context, match_context)],
            'output_layer',
            'output_layer')

    def model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')

        #input layer
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs
        
        # attention layer
        att_context = self.gated_attention_gru_layer(c_processed, q_processed)

        # seif-matching_attention layer
        match_context = self.matching_attention_layer(att_context)

        # output layer
        start_logits, end_logits = self.output_layer(q_processed, match_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        #paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        return C.combine([start_logits, end_logits]), new_loss
