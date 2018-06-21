import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from collections import OrderedDict
from torch.nn import functional as F
from layers import l2norm
from layers import global_initializer


class SelfAttentiveEncoder(nn.Module):
    '''
        Self-attention module
    '''
    def __init__(self, nb_features=300, att_units=300, hops=30):
        super(SelfAttentiveEncoder, self).__init__()

        self.drop = nn.Dropout(0.0)
        self.ws1 = nn.Linear(nb_features, att_units, bias=False)
        self.ws2 = nn.Linear(att_units, hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()       
        self.attention_hops = hops        

    def forward(self, inp, zero_mask=None):
        
        size = inp.size()  # [batch, len, nhid]       
        x = inp
        hbar = self.tanh(self.ws1(self.drop(x)))  # [batch*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [batch, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [batch, hop, len]

        if zero_mask is not None:
            # ignores zero padding
            alphas = alphas + (
                -10000 * zero_mask.float().unsqueeze(1))
            # [batch, hop, len] + [batch, hop, len]
        
        alphas = self.softmax(alphas.view(-1, size[1]))  # [batch*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [batch, hop, len]
        return torch.bmm(alphas, inp), alphas


class GRUAttentiveTextEncoder(nn.Module):
    '''
        SEAM-G
    '''
    def __init__(self, vocab_size, word_dim, embed_size,
                 use_abs=False, att_units=200, hops=15, 
                 gru_units=1024, num_layers=1, norm_words=None):

        super(GRUAttentiveTextEncoder, self).__init__()

        self.use_abs = use_abs
        self.embed_size = embed_size
        self.hops = hops 
        self.att_units = att_units

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.norm_words = norm_words
        # caption embedding
        self.rnn = nn.GRU(word_dim, gru_units, num_layers, batch_first=True)

        self.attention = SelfAttentiveEncoder(nb_features=gru_units, 
            att_units=att_units, hops=hops)

        self.fc = nn.Linear(gru_units * hops, embed_size)
        global_initializer(self)

    def forward(self, inputs, lengths):
        
        # Embed word ids to vectors
        x = self.embed(inputs)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)[0]
        

        out, att_weights = self.attention(padded, (inputs == 0))
        self.attention_weights = att_weights
        out = out.view(inputs.size()[0], -1)
        
        fc_out = self.fc(out)

        # normalization in the joint embedding space
        outnormed = l2norm(fc_out)
    
        # take absolute value, used by order embeddings
        if self.use_abs:
            outnormed = torch.abs(outnormed)

        return outnormed


class ConvAttentiveTextEncoder(nn.Module):
    '''
        SEAM-C
    '''
    def __init__(self, vocab_size, word_dim, embed_size,
                 use_abs=False, att_units=300, hops=30, 
                 gru_units=None, num_layers=1, norm_words=None):
        super(ConvAttentiveTextEncoder, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.hops = hops 
        self.att_units = att_units

        conv_filters = 100
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.conv1 = ConvBlock(
                        in_channels=word_dim, 
                        out_channels=conv_filters, 
                        kernel_size=2, 
                        padding=1,                        
                        activation='ReLU',
                        batchnorm=True,)

        self.conv2 = ConvBlock(
                        in_channels=word_dim, 
                        out_channels=conv_filters, 
                        kernel_size=3, 
                        padding=1,                        
                        activation='ReLU',
                        batchnorm=True,)

        self.att_emb = SelfAttentiveEncoder(nb_features=word_dim, 
            att_units=att_units, hops=hops)

        self.att_conv1 = SelfAttentiveEncoder(nb_features=conv_filters, 
            att_units=att_units, hops=hops)

        self.att_conv2 = SelfAttentiveEncoder(nb_features=conv_filters, 
            att_units=att_units, hops=hops)

        self.fc = nn.Linear((conv_filters * 2 + word_dim) * hops, embed_size)
        global_initializer(self)    

    def forward(self, inputs, lengths):
        
        # Embed word ids to vectors
        x_embed = self.embed(inputs)
        x = x_embed.permute(0, 2, 1) # [B, F, T]
                
        conv1 = self.conv1(x)[:,:,:-1]
        conv1a, conv1a_vis = self.att_conv1(conv1.permute(0, 2, 1),
                                             (inputs == 0)) # 10 * 100 = 1000
        conv1a = conv1a.view(conv1a.size()[0], -1)

        conv2 = self.conv2(x)
        conv2a, conv2a_vis = self.att_conv2(conv2.permute(0, 2, 1), 
                                            (inputs == 0)) # 10 * 100 = 1000
        conv2a = conv2a.view(conv2a.size()[0], -1)

        emb_att, emb_vis = self.att_emb(x_embed, (inputs == 0)) # 10 * 300 = 3000
        self.attention_weights = emb_att

        emb_att = emb_att.view(emb_att.size()[0], -1)

        vectors = torch.cat([conv1a, conv2a, emb_att], 1) # [B, 5000]
                
        fc_out = self.fc(vectors)

        # normalization in the joint embedding space
        outnormed = l2norm(fc_out)
    
        # take absolute value, used by order embeddings
        if self.use_abs:
            outnormed = torch.abs(outnormed)       

        return outnormed        


class AttentiveTextEncoder(nn.Module):
    '''
        SEAM-E
    '''
    def __init__(self, vocab_size, word_dim, embed_size,
                 use_abs=False, att_units=300, hops=30, 
                 gru_units=None, num_layers=None, norm_words=False):
        super(AttentiveTextEncoder, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.hops = hops
        self.att_units = att_units

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        
        self.attention = SelfAttentiveEncoder(nb_features=word_dim, 
            att_units=att_units, hops=hops)

        att_out_size = word_dim * hops

        self.fc = False
        if att_out_size != embed_size:
            self.fc = nn.Linear(att_out_size, embed_size)

        global_initializer(self)

    def forward(self, inputs, lengths):        
        
        # Embed word ids to vectors
        x = self.embed(inputs)
       
        out, att_weights = self.attention(x, (inputs == 0))
        self.attention_weights = att_weights
        out = out.view(inputs.size()[0], -1)
        
        if self.fc:
            out = self.fc(out)            

        # normalization in the joint embedding space
        outnormed = l2norm(out)
    
        # take absolute value, used by order embeddings
        if self.use_abs:
            outnormed = torch.abs(outnormed)

        return outnormed

# tutorials/08 - Language Model
# RNN Based Language Model
class GRUTextEncoder(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, gru_units=1024):
        super(GRUTextEncoder, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.gru_units = gru_units

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, gru_units, num_layers, batch_first=True)
        
        self.fc = None
        if embed_size != gru_units:
            self.fc = nn.Linear(gru_units, embed_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

        if self.fc:
            r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)

            self.fc.weight.data.uniform_(-r, r)
            self.fc.bias.data.fill_(0)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.gru_units)-1)
        if torch.cuda.is_available():
            I = I.cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        if self.fc:
            out = self.fc(out)
        
        # normalization in the joint embedding space
        outnormed = l2norm(out)
        # take absolute value, used by order embeddings
        if self.use_abs:
            outnormed = torch.abs(outnormed)

        return outnormed


class ConvBlock(nn.Module):

    def __init__(self, activation=None, batchnorm=False, **kwargs):
        super(ConvBlock, self).__init__()
        
        layers = OrderedDict()
        layers['conv'] = nn.Conv1d(**kwargs)        
        if activation is not None:
            layers['activation'] = eval('nn.{}'.format(activation))()

        if batchnorm:
            layers['bn'] = nn.BatchNorm1d(kwargs['out_channels'])

        self.conv = nn.Sequential(layers)

    def forward(self, x):
        return self.conv(x)


text_encoders_alias = {
    'gru': {'method': GRUTextEncoder, 'args': {}},
    'seam-e': {'method': AttentiveTextEncoder, 'args': {}},
    'seam-g': {'method': GRUAttentiveTextEncoder, 'args': {}},
    'seam-c': {'method': ConvAttentiveTextEncoder, 'args': {}},    
}


def get_text_encoder(encoder, opt):

    encoder = encoder.lower()
    vocab_size = opt.vocab_size
    word_dim = opt.word_dim
    num_layers = opt.num_layers
    embed_size = opt.embed_size
    use_abs = opt.use_abs

    try:
        gru_units = opt.gru_units
        norm_words = opt.norm_words
    except AttributeError:
        gru_units = embed_size
        norm_words = None

   
    params = {
        'vocab_size': vocab_size, 
        'word_dim': word_dim,
        'gru_units': gru_units,             
        'embed_size': embed_size, 
        'num_layers': num_layers,
        'use_abs': opt.use_abs,            
    }

    if encoder.startswith('seam'):
        params['att_units'] = opt.att_units 
        params['hops'] = opt.att_hops

    try:
        txt_enc = eval(encoder)(**params)
    except NameError:
        params.update(text_encoders_alias[encoder]['args'])
        txt_enc = text_encoders_alias[encoder]['method'](**params)
    return txt_enc