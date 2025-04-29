import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Subset

class BilingualDataset(Dataset):
    
    def __init__(self, opt, ds, tokenizer_src, tokenizer_tgt):
        super(BilingualDataset, self).__init__()

        self.opt = opt
        self.max_samples = opt.input.number_samples
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = opt.input.seq_len
        self.src_lang = opt.input.lang_src
        self.tgt_lang = opt.input.lang_tgt

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")]).long()
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")]).long()
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")]).long()

        # Select the first `max_samples` elements if provided
        if self.max_samples is not None:
            self.ds = Subset(ds, list(range(min(self.max_samples, len(ds)))))
        else:
            self.ds = ds

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Tokenization
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1  # -1 for EOS

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Input sequence too long")
        
        # Create padded encoder input
        encoder_input = torch.cat([
            self.sos_token, 
            torch.Tensor(enc_input_tokens).long(), 
            self.eos_token, 
            torch.Tensor([self.pad_token] * enc_num_padding).long()
        ]).long()
        
        # Create decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(dec_input_tokens).long(),
            torch.Tensor([self.pad_token] * dec_num_padding).long()
        ]).long()

        # Create label (target output)
        label = torch.cat([
            torch.Tensor(dec_input_tokens).long(),
            self.eos_token,
            torch.Tensor([self.pad_token] * dec_num_padding).long()
        ]).long()

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).int()
    return mask == 0 # Allowing only past tokens 


        