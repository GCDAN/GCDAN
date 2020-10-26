import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator
		self.eps = 1e-8
		self.sim = None

	def forward(self, src, tgt, src_mask, tgt_mask, uid):
		"Take in and process masked src and target sequences."	
		m = self.encode(src, src_mask)
		om = self.encode(tgt, tgt_mask)
		'''
		src_readout = F.normalize(torch.bmm(src_mask.float(), m).squeeze(1), dim = -1)
		tgt_readout = F.normalize(torch.matmul(tgt_mask.float(), om), dim = -1)
		sim = torch.matmul(tgt_readout, src_readout.transpose(-2,-1))
		sim = F.softmax(sim, dim = -1)
		'''
		src_readout = torch.bmm(F.normalize(src_mask.float(),dim=-1,p=1), m)
		src_readout = src_readout.squeeze(1).unsqueeze(0).unsqueeze(0).repeat(om.size(0),om.size(1),1,1)
		tgt_readout = torch.matmul(F.normalize(tgt_mask.float(),dim=-1,p=1), om)
		tgt_readout = tgt_readout.unsqueeze(2).repeat(1,1,m.size(0),1)
		#print(src_readout.size())
		#print(tgt_readout.size())
		sim = src_readout - tgt_readout
		sim = sim.pow(2)
		sim = sim.sum(dim = -1)
		sim = sim.sqrt()
		sim = torch.exp(-sim)
		sim = F.normalize(sim,p=1,dim = -1)
		self.sim = sim
		o, ot_emb = self.decode(m, src_mask, tgt, tgt_mask, sim)
		y = self.generator(o, uid, ot_emb)
		return y
		#return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)

	def encode(self, src, src_mask):
		emb = self.src_embed(src)
		return self.encoder(emb[0], src_mask)
		#return self.encoder(src, src_mask)
	
	def decode(self, memory, src_mask, tgt, tgt_mask, sim):
		emb = self.tgt_embed(tgt)
		return (self.decoder(emb[0], memory, src_mask, tgt_mask, sim), emb[1])
		#return self.decoder(tgt, memory, src_mask, tgt_mask)
class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab, uid_size, uid_emb_size):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)
		self.emb_uid = nn.Embedding(uid_size, uid_emb_size)

	def forward(self, x, uid, ot):
		uid_emb = self.emb_uid(torch.LongTensor([uid]))
		#print(uid_emb.size())
		uid_emb = uid_emb.unsqueeze(0).repeat(x.size(0),x.size(1),1)
		#x = torch.cat((x, uid_emb), dim = -1)
		#x = torch.cat((x, ot), dim = -1)
		return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
	"Core encoder is a stack of N layers"
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, mask):
		"Pass the input (and mask) through each layer in turn."
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)
class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		x = x.float()
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))
		#return self.norm(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask, sim):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask, sim)
		return self.norm(x)
class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask, sim):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, sim))
		return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	'''
	print('query:'+str(query.size()))
	print('key:'+str(key.size()))
	print('value:'+str(value.size()))
	print('mask:'+str(mask.size()))
	'''
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	
	return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):

		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		#print(query)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
		'''
		print('report...')
		print(query.size())
		print(key.size())
		print(value.size())
		'''
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)
class InterMultiHeadAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(InterMultiHeadAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None, sim = None):
		'''
		# normal attention
		q_batches = query.size(0)
		k_batches = key.size(0)
		query = self.linears[0](query).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		key = key.view(1, -1, key.size(-1)).repeat(q_batches,1,1)
		key = self.linears[1](key).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		value = value.view(1, -1, value.size(-1)).repeat(q_batches,1,1)
		src_mask = mask.view(1, -1).unsqueeze(1).repeat(q_batches,1,1,1)
		value = self.linears[2](value).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		x, self.attn = attention(query, key, value, mask=src_mask, 
								 dropout=self.dropout)
		x = x.transpose(1, 2).contiguous() \
			 .view(q_batches, -1, self.h * self.d_k)
		return self.linears[-1](x)
		'''
		
		#print('mask:' + str(mask.size()))
		#print('query:' + str(query.size()))
		#print('key:' + str(key.size()))
		#print('value:' + str(value.size()))
		# 0.1) read_out
		q_batches = query.size(0)
		k_batches = key.size(0)

		#print('read_out:' + str(read_out.size()))
		# 0.2) calculate similarity
		#problem 2 cosine similarity
		if sim is not None:
			sim = sim.unsqueeze(-1).repeat(1,1,1,key.size(1))
			sim = sim.view(q_batches, query.size(1),-1)
			sim = sim.unsqueeze(1).repeat(1,self.h,1,1)
		#print('sim:' + str(sim.size()))

		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query = self.linears[0](query).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		key = key.view(1, -1, key.size(-1)).repeat(q_batches,1,1)
		key = self.linears[1](key).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		value = value.view(1, -1, value.size(-1)).repeat(q_batches,1,1)
		value = self.linears[2](value).view(q_batches, -1, self.h, self.d_k).transpose(1, 2)
		#print('query:' + str(query.size()))
		#print('key:' + str(key.size()))
		#print('value:' + str(value.size()))
		#query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
		# 2) Apply attention on all the projected vectors in batch. 
		src_mask = mask.view(1, -1).unsqueeze(1).repeat(q_batches,1,1,1)
		#print('src_mask:' + str(src_mask.size()))
		x, self.attn = traj_attention(query, key, value, mask=src_mask, sim = sim, dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(q_batches, -1, self.h * self.d_k)
		#print('x:' + str(x.size()))
		return self.linears[-1](x)
		

		
def traj_attention(query, key, value, mask=None, sim = None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	'''
	print('query:'+str(query.size()))
	print('key:'+str(key.size()))
	print('value:'+str(value.size()))
	print('mask:'+str(mask.size()))
	print('sim:'+str(sim.size()))
	'''
	if sim is not None:
		scores = torch.mul(sim, scores)
		#scores = scores + sim
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	
	return torch.matmul(p_attn, value), p_attn
class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		res = self.lut(x) * math.sqrt(self.d_model)
		return self.lut(x) * math.sqrt(self.d_model)

class Embeddings_traj(nn.Module):
	def __init__(self, d_model, vocab, g = None):
		super(Embeddings_traj, self).__init__()
		self.lut_loc = nn.Linear(vocab[0], d_model[0])
		#self.lut_loc = nn.Embedding(vocab[0], d_model[0])
		self.loc_size = vocab[0]
		self.lut_st = nn.Embedding(vocab[1], d_model[1])
		self.st_size = vocab[1]
		self.lut_ed = nn.Embedding(vocab[2], d_model[2])
		self.ed_size = vocab[2]
		self.d_model = sum(d_model)
		self.g = g
		#self.d_model = d_model[0]
		if self.g is not None:
			_A = self.g + torch.eye(self.loc_size)
			deg = sum(_A)
			_d = torch.diag(torch.pow(deg , -0.5))
			_A = _A.cuda()
			_d = _d.cuda()
			self.A = torch.matmul(torch.matmul(_d,_A), _d)

	def forward(self, x):
		loc, st, ed = x
		yt = torch.cat((st[:,1:], torch.zeros((st.size(0),1), dtype = torch.long)), -1)
		#print(one_hot.size())
		#print(self.g.size())
		
		if self.g is not None:
			one_hot = F.one_hot(loc, self.loc_size).float()
			one_hot = one_hot.cuda()
			loc = torch.matmul(one_hot, self.A)
			loc = loc.cpu()
		res_loc = F.relu(self.lut_loc(loc))
		'''
		res_loc = self.lut_loc(loc)
		'''
		res = res_loc
		res_st = self.lut_st(st)
		res_yt = self.lut_st(yt)
		res_ed = self.lut_ed(ed)
		res = torch.cat((res, res_st), -1)
		res = torch.cat((res, res_ed), -1)
		#print(res.size())
		return (res * math.sqrt(self.d_model), res_yt)
		#res = self.lut(x) * math.sqrt(self.d_model)
		#return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=300):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		te = x[1]
		x = x[0]
		#print(x.size())
		#print(t.size())
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		#x = x + Variable(self.pe[:, t], requires_grad=False)
		return (self.dropout(x), te)

class TrajTransformer(nn.Module):
	def __init__(self, parameters, graph = None):
		super(TrajTransformer, self).__init__()
		self.loc_size = parameters.loc_size+1
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size+1
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		input_size = self.loc_emb_size + 2 * self.tim_emb_size
		#input_size = self.loc_emb_size
		N = 3
		d_ff = 1024
		h = 4
		c = copy.deepcopy
		attn = MultiHeadedAttention(h, input_size)
		inter_attn = InterMultiHeadAttention(h, input_size)
		ff = PositionwiseFeedForward(input_size, d_ff, parameters.dropout_p)
		position = PositionalEncoding(input_size, parameters.dropout_p)
		emb = nn.Sequential(Embeddings_traj((self.loc_emb_size, self.tim_emb_size, self.tim_emb_size), (self.loc_size, self.tim_size, self.tim_size), g = graph), c(position))
		self.model = EncoderDecoder(
		Encoder(EncoderLayer(input_size, c(attn), c(ff), parameters.dropout_p), N),
		Decoder(DecoderLayer(input_size, c(attn), c(inter_attn), 
							 c(ff), parameters.dropout_p), N),
		emb, emb, Generator(input_size, self.loc_size, self.uid_size, self.uid_emb_size))
		for p in self.model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform(p)
	def forward(self, src_loc, src_st, src_ed, tgt_loc, tgt_st, tgt_ed, target_len, uid):
		src_mask = (src_loc != 0).unsqueeze(-2)
		tgt_mask = (tgt_loc != 0).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(tgt_loc.size(-1)).type_as(tgt_mask.data))
		#print(src_loc.size())
		return self.model((src_loc, src_st, src_ed),(tgt_loc, tgt_st, tgt_ed),src_mask ,tgt_mask, uid)
def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0   
