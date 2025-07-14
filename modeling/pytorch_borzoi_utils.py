#MIT License
#
#Copyright (c) 2021 Phil Wang

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
# =========================================================================
from flash_attn.ops.layer_norm import dropout_add_layer_norm
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    
class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def poisson_multinomial_torch(
        y_pred,
        y_true,
        total_weight: float = 0.2,
        epsilon: float = 1e-6,
        rescale: bool = False,
    ):
        """Possion decomposition with multinomial specificity term.

        Args:
          total_weight (float): Weight of the Poisson total term.
          epsilon (float): Added small value to avoid log(0).
        """
        seq_len = y_true.shape[1]


        # add epsilon to protect against tiny values
        y_true += epsilon
        y_pred += epsilon

        # sum across lengths
        s_true = y_true.sum(dim = 1, keepdim=True)
        s_pred = y_pred.sum(dim = 1, keepdim=True)

        # normalize to sum to one
        p_pred = y_pred / s_pred

        # total count poisson loss
        poisson_term = F.poisson_nll_loss(s_pred,s_true, log_input = False, eps = 0,reduction = 'mean')  # B x T
        #print (poisson_term,poisson_term.shape)
        poisson_term /= seq_len
        #print (poisson_term)

        # multinomial loss
        pl_pred = torch.log(p_pred)  # B x L x T
        multinomial_dot = -torch.multiply(y_true, pl_pred)  # B x L x T
        multinomial_term = multinomial_dot.sum(dim = 1)  # B x T
        multinomial_term /= seq_len

        # normalize to scale of 1:1 term ratio
        loss_raw = multinomial_term + total_weight * poisson_term
        if rescale:
            loss_rescale = loss_raw * 2 / (1 + total_weight)
        else:
            loss_rescale = loss_raw

        return loss_rescale.mean()


def add_flashzoi_weight_decay(model, weight_decay=1e-5, weight_decay_transformer = 1e-8, skip_list=()):
        decay = []
        decay_transformer = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list or 'bias' in name:
                accelerator.print ('No decay', name)
                no_decay.append(param)
            elif 'freq' in name:
                accelerator.print ('No decay', name)
                no_decay.append(param)
            elif 'transformer' in name:
                accelerator.print ('Decay Transformer:', name)
                decay_transformer.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}, {'params': decay_transformer, 'weight_decay': weight_decay_transformer}]
