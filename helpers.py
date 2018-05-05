import numpy as np
import cntk as C
from cntk.layers.blocks import _INFERRED

def OptimizedRnnStack(hidden_dim, num_layers=1, recurrent_op='gru', bidirectional=False, use_cudnn=True, name=''):
    if use_cudnn:
        W = C.parameter(_INFERRED + (hidden_dim,), init=C.glorot_uniform())
        def func(x):
            return C.optimized_rnnstack(x, W, hidden_dim, num_layers, bidirectional, recurrent_op=recurrent_op, name=name)
        return func
    else:
        def func(x):
            return C.splice(
                        C.layers.Recurrence(C.layers.GRU(hidden_dim))(x),
                        C.layers.Recurrence(C.layers.GRU(hidden_dim), go_backwards=True)(x),
                        name=name)
        return func
    
def seq_loss(logits, y):
    prob = C.sequence.softmax(logits)
    return -C.log(C.sequence.last(C.sequence.gather(prob, y)))

'''
def attention_pooling(inputs, inputs_weights, decode, decode_weights, keys):
    """
    inputs: shape=(dim, n)
    inputs_weight: shape=(dim, dim)
    decode: shape=(1, dec_dim)
    decode_weights: shape=(dim, dec_dim)
    keys: shape=(1, dim)
    
    """
    w_in = C.times(inputs_weights ,inputs)  #shape=(dim, n)
    decode = C.transpose(decode, perm=(1,0))
    w_dec = C.times(decode_weights ,decode) #shape=(dim, dim)
    S = C.tanh(C.plus(C.transpose(w_in, perm=(1,0)), C.transpose(w_dec, perm=(1,0)))) #shape=(n, dim)
    S = C.times(S, C.transpose(keys, perm=(1,0))) #shape=(n)
    S = C.ops.sequence.softmax(S, name="softmax")
    attention = C.transpose(C.times(inputs ,S), perm=(1,0))
    return attention
'''  
def attention_pooling(inputs, inputs_mask, inputs_weights, decode, decode_weights, keys):
    """
    inputs: shape=(n, dim)
    inputs_weight: shape=(dim, dim)
    decode: shape=(1, dec_dim)
    decode_weights: shape=(dec_dim, dim)
    keys: shape=(dim, 1)
    
    """
    w_in = C.times(inputs, inputs_weights)  #shape=(n, dim)
    w_dec = C.times(decode, decode_weights) #shape=(dim, 1)
    S = C.tanh(w_in + C.sequence.broadcast_as(w_dec, w_in)) #shape=(n, dim)
    S = C.element_select(inputs_mask, S, C.constant(-1e+30))
    S = C.times(S, keys) #shape=(n)
    S = C.ops.sequence.softmax(S, name="softmax")
    attention = C.reduce_sum(inputs * S, axis=0)
    return attention

'''
def question_pooling(inputs, inputs_dim):
    Wp = C.parameter(shape=(inputs_dim,inputs_dim))
    Vp = C.parameter(shape=(inputs_dim, 1))
    outputs_w = C.times(C.tanh(C.times(inputs, Wp)), Vp)
#    Vp = C.parameter(shape=(inputs_dim))
#    outputs_w = C.sequence.reduce_sum(C.tanh(C.times(inputs, Wp)) * Vp, 1)
    
    outputs_w = C.sequence.softmax(inputs)
    outputs = outputs_w * inputs
    return outputs

def att_weight(h_enc, h_dec, inputs_dim):
    w_enc = C.parameter(shape=(inputs_dim,inputs_dim))
    w_dec = C.parameter(shape=(inputs_dim,inputs_dim))
    wh_enc = C.times(h_enc, w_enc)
    wh_dec = C.times(h_dec, w_dec)
    s_t = C.tanh(wh_dec + wh_enc)
    v_t = C.parameter(shape=(inputs_dim, 1))
    s_t = C.times(s_t ,v_t)
#    v_t = C.parameter(shape=(inputs_dim))
#    s_t = C.sequence.reduce_sum(s_t * v_t, 1)    
    
    wh_weight = C.sequence.softmax(s_t)
    return wh_weight  
'''
'''
def question_pooling(inputs, inputs_dim):
    inputs_w, inputs_mask = C.sequence.unpack(inputs, padding_value=0).outputs
    Wp = C.parameter(shape=(inputs_dim,inputs_dim))
    Vp = C.parameter(shape=(inputs_dim,1))
    outputs_w = C.times(C.tanh(C.times(inputs_w, Wp)), Vp)
    outputs_w = C.softmax(C.element_select(inputs_mask, outputs_w, C.constant(-1e+30)), axis=0)
    outputs = outputs_w * inputs_w
    outputs = C.reduce_sum(outputs, 0)
    return outputs
   
def att_weight(h_enc, h_dec, inputs_dim):
    h_enc_w, h_enc_mask = C.sequence.unpack(h_enc, padding_value=0).outputs
    w_enc = C.parameter(shape=(inputs_dim, inputs_dim))
    w_dec = C.parameter(shape=(inputs_dim, inputs_dim))
    v_t = C.parameter(shape=(inputs_dim))
    wh_enc = C.times(h_enc_w, w_enc)
    wh_dec = C.times(h_dec, w_dec)
    s_t = C.tanh(C.sequence.broadcast_as(wh_dec, wh_enc) + wh_enc)
    s_t = C.element_select(h_enc_mask, s_t, C.constant(-1e+30))
    s_t = C.reduce_sum(s_t * v_t, 1)
    wh_weight = C.softmax(s_t)
    return wh_weight    
'''
def question_pooling(inputs, inputs_dim):
    outputs_w = C.layers.Dense(1, activation=C.tanh, name='out_start')(inputs)
    outputs_w = C.sequence.softmax(outputs_w)
    outputs = C.sequence.reduce_sum(outputs_w * inputs)
    return outputs

def attention_weight(h_enc, h_dec, inputs_dim):
    enc = C.layers.Dense(inputs_dim, name='out_start')(h_enc)
    dec = C.sequence.broadcast_as(C.layers.Dense(inputs_dim, name='out_start')(h_dec), enc)
    att_weight = C.layers.Dense(1, name='out_start')(C.tanh(enc+dec))
    att_weight = C.sequence.softmax(att_weight)
    return att_weight      

def all_spans_loss(start_logits, start_y, end_logits, end_y):
    # this works as follows:
    # let end_logits be A, B, ..., Y, Z
    # let start_logits be a, b, ..., y, z
    # the tricky part is computing log sum (i<=j) exp(start_logits[i] + end_logits[j])
    # we break this problem as follows
    # x = logsumexp(A, B, ..., Y, Z), logsumexp(B, ..., Y, Z), ..., logsumexp(Y, Z), Z
    # y = a + logsumexp(A, B, ..., Y, Z), b + logsumexp(B, ..., Y, Z), ..., y + logsumexp(Y, Z), z + Z
    # now if we exponentiate each element in y we have all the terms we need. We just need to sum those exponentials...
    # logZ = last(sequence.logsumexp(y))
    x = C.layers.Recurrence(C.log_add_exp, go_backwards=True, initial_state=-1e+30)(end_logits)
    y = start_logits + x
    logZ = C.layers.Fold(C.log_add_exp, initial_state=-1e+30)(y)
    return logZ - C.sequence.last(C.sequence.gather(start_logits, start_y)) - C.sequence.last(C.sequence.gather(end_logits, end_y))

def seq_hardmax(logits):
    seq_max = C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30, logits.shape))(logits)
    s = C.equal(logits, C.sequence.broadcast_as(seq_max, logits))
    s_acc = C.layers.Recurrence(C.plus)(s)
    return s * C.equal(s_acc, 1) # only pick the first one

class LambdaFunc(C.ops.functions.UserFunction):
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg: print((len(arg), arg[0].shape,) if type(arg) == list else (1, arg.shape,), arg),
            name=''):
        self.when = when
        self.execute = execute

        super(LambdaFunc, self).__init__([arg], name=name)

    def infer_outputs(self):
        return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        if self.when(argument):
            self.execute(argument)

        return None, argument

    def backward(self, state, root_gradients):
        return root_gradients
        
    def clone(self, cloned_inputs):
        return self.__init__(*cloned_inputs)
        
def print_node(v):
    return C.user_function(LambdaFunc(v))