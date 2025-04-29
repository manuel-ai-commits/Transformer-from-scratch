### SEACH FOR "..." on every file to see where you need to add your code ###
from src import utils

import math
import os

import math
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms

import torchmetrics # Pytorch Lightning Metrics for text generation

class InputEmbeddings(nn.Module):
    def __init__(self, opt, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.opt = opt
        self.vocab_size = vocab_size
        self.d_model = opt.model.d_model
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embeddings(x)* math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, opt, seq_len: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.opt = opt 
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=self.opt.model.dropout)    

        # matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.opt.model.d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1).float() # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.opt.model.d_model, 2).float() * (-math.log(10000.0) / self.opt.model.d_model)) # (d_model/2)
        # Sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # (1, seq_len, d_model)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, opt):
        super(LayerNorm, self).__init__()
        self.opt = opt
        self.eps = self.opt.model.eps
        self.features = opt.model.d_model
        self.alpha = nn.Parameter(torch.ones(self.features)) # multiplied
        self.beta = nn.Parameter(torch.zeros(self.features)) # added

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForward(nn.Module):

    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.linear_1 = nn.Linear(opt.model.d_model, opt.model.d_ff) # w1 and b1
        self.dropout = nn.Dropout(opt.model.dropout)
        self.linear_2 = nn.Linear(opt.model.d_ff, opt.model.d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

    
class MultiHeadAttention(nn.Module):

    def __init__(self, opt):
        super(MultiHeadAttention, self).__init__()
        self.opt = opt
        self.d_model = opt.model.d_model
        self.h = opt.model.n_heads
        assert self.d_model % self.h == 0, "d_model is divisible by h"

        self.d_k = self.d_model // self.h
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        self.W_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(p=opt.model.dropout)

    @staticmethod
    def attention(q, k, v, mask=None, dropout=False):
        d_k = q.size(-1)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9) # if mask is 0, fill with -1e9
        
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len); -1e9 will be close to 0

        if dropout:
            attention_scores = dropout(attention_scores)

        return attention_scores @ v, attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.W_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.W_k(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.W_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # Split the d_model into h heads
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)

        # Attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.opt.model.if_dropout) # (batch_size, h, seq_len, d_k)    


        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)

        return self.W_o(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    
class ResidualConnection(nn.Module):

    def __init__(self, opt):
        super(ResidualConnection, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(p=opt.model.dropout)
        self.norm = LayerNorm(opt)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, opt):
        super(EncoderBlock, self).__init__()
        self.opt = opt
        self.self_attention = MultiHeadAttention(opt)
        self.feed_forward = FeedForward(opt)
        self.residual_connection = nn.ModuleList([ResidualConnection(opt) for _ in range(2)])

    def forward(self, x, src_mask): # src_mask is the mask for the source sequence, no interaction between padding and actual words

        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask)) # calling the forward method of multiheadattention
        x = self.residual_connection[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList([EncoderBlock(opt) for _ in range(opt.model.n_layers_enc)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, opt):
        super(DecoderBlock, self).__init__()
        self.opt = opt
        self.self_attention = MultiHeadAttention(opt)
        self.cross_attention = MultiHeadAttention(opt)
        self.feed_forward = FeedForward(opt)
        self.residual_connection = nn.ModuleList([ResidualConnection(opt) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): # src_mask is the mask for the source sequence original language, the tgt_mask is the mask for the target sequence, the language to be translated to
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList([DecoderBlock(opt) for _ in range(opt.model.n_layers_dec)])
        self.LayerNorm = LayerNorm(opt)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.LayerNorm(x)


class Projection(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Projection, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(opt.model.d_model, vocab_size)
        
    def forward(self, x):

        return F.log_softmax(self.fc(x), dim=-1) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
    

class Transformer_setup(nn.Module):
    def __init__(self, opt, src_vobab_size, tgt_vobab_size, src_seq_len, tgt_seq_len):
        super().__init__()
        self.opt = opt
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.src_embed = InputEmbeddings(opt, src_vobab_size)
        self.tgt_embed = InputEmbeddings(opt, tgt_vobab_size)
        self.src_pos = PositionalEncoding(opt, src_seq_len)
        self.tgt_pos = PositionalEncoding(opt, tgt_seq_len)
        self.proj = Projection(opt, tgt_vobab_size)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def projection(self, x):
        return self.proj(x)
    



class Transformer(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(Transformer, self).__init__()

        self.opt = opt
        self.device = opt.device

        if self.device == "mps":
            torch.set_num_threads(8)

        
        # Initial settings
        self.batch_size = self.opt.input.batch_size
        self.dataset = self.opt.input.dataset

        # Model settings
        self.d_model = self.opt.model.d_model
        self.d_ff = self.opt.model.d_ff
        self.h = self.opt.model.n_heads
        self.dropout = self.opt.model.dropout
        
        self.seq_len = self.opt.input.seq_len

        self.tokenizer_src, self.tokenizer_tgt = utils.get_data_or_tokenizer(self.opt, "train", data=False)
        self.src_vocab_size = self.tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        
        # Create the transformer
        self.model = Transformer_setup(opt, self.src_vocab_size, self.tgt_vocab_size, self.seq_len, self.seq_len)

        # Initialize the parameters
        

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)

        self._init_weights()

        

    def _init_weights(self):
        
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    
    def forward(self, inputs, scalar_outputs=None):
        if scalar_outputs is None:
            scalar_outputs = {"Loss": torch.zeros(1, device=self.opt.device)}
        torch.autograd.set_detect_anomaly(True)
        
        z = inputs
        
        encoder_input = z["encoder_input"] # (batch_size, seq_len)
        decoder_input = z["decoder_input"] # (batch_size, seq_len)
        encoder_mask = z["encoder_mask"]   # (batch_size, 1, 1, seq_len)
        decoder_mask = z["decoder_mask"]   # (batch_size, 1, seq_len, seq_len)
        

        encoder_output = self.model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
        decoder_output = self.model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
        proj_output = self.model.projection(decoder_output) # (batch_size, seq_len, vocab_size)
        # print(encoder_output.shape)
        # print(decoder_mask.shape)
        # print(decoder_output.shape)
        # print(proj_output.shape)

        label = z["label"]                 # (batch_size, seq_len)

        loss = self.loss_fn(proj_output.view(-1, self.tgt_vocab_size), label.view(-1)) # (batch_size * seq_len, vocab_size), (batch_size * seq_len)
        scalar_outputs["Loss"] += loss
            
        return scalar_outputs

        
    

    def predict(self, inputs, visualize=False, scalar_outputs=None, num_generate=25, grid_size=0.05):
        if scalar_outputs is None:
            scalar_outputs = {
                "CER": torch.zeros(1, device=self.opt.device),
                "WER": torch.zeros(1, device=self.opt.device),
                "BLEU": torch.zeros(1, device=self.opt.device)
            }
        z = inputs

        sos_idx = self.tokenizer_src.token_to_id("[SOS]")
        eos_idx = self.tokenizer_src.token_to_id("[EOS]")
    
        encoder_input = z["encoder_input"] # (batch_size, seq_len)
        encoder_mask = z["encoder_mask"]   # (batch_size, 1, 1, seq_len)
        src_text = z["src_text"]
        tgt_text = z["tgt_text"]

        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        

        # Reconstruction task
        decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(self.device) # (1, 1)
        with torch.no_grad():
            encoder_output = self.model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)

            while True:
                if decoder_input.size(1) >= self.seq_len:
                    break
            
                # Build mask to avoid watching future words
                decoder_mask = utils.casual_mask(decoder_input.size(1)).type_as(encoder_mask).to(self.device) # (1, seq_len, seq_len)

                # Calculate the next word
                decoder_output = self.model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (1, seq_len, d_model)

                # Get the next token
                prob = self.model.projection(decoder_output[:,-1])
                # Select token with max probability
                _ , next_word = torch.max(prob, dim=1)
                # print("next_word", next_word)
                # print(encoder_output.shape)
                # print(decoder_mask.shape)
                # print(decoder_output.shape)
                # print(prob.shape)

                # Add the next word to the sequence
                decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(encoder_input).fill_(next_word.item()).to(self.device)], dim=1)

                if next_word.item() == eos_idx:
                    break
            
        decoder_input = decoder_input.squeeze(0)
        
        src = src_text[0]
        tgt = tgt_text[0]
        pred = self.tokenizer_tgt.decode(decoder_input.detach().cpu().numpy())
        if visualize:
            print(f"SRC: {src}\nTGT: {tgt}\nPRED: {pred}\n")
        
        cer = torchmetrics.text.CharErrorRate()
        wer = torchmetrics.text.WordErrorRate()
        bleu = torchmetrics.text.BLEUScore()

        scalar_outputs["CER"] += cer(pred, tgt)
        scalar_outputs["WER"] += wer(pred, tgt)
        scalar_outputs["BLEU"] += bleu(pred, tgt)

        return scalar_outputs
        


        