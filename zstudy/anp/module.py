import torch as t 
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear',*args, **kwargs):
        '''
        w_init : weight initialization을 어떻게 할지를 결정 
        '''
        # super(Linear, self).__init__(*args, **kwargs)
        super().__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)


        # Linear layer의 weight를 Xavier Uniform Initialization으로 초기화 
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain = nn.init.calculate_gain(w_init)
        )

    # 실제 데이터가 layer를 통과할 때 무슨 일을 할지 정의 
    def forward(self,x):
        return self.linear_layer(x)

# (x,y)를 입력 받아서 latent variable z를 만들기 위한 분포의 파라미터 mu, log_sigma를 계산하는 encoder 
class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim=3, *args, **kwargs):
        '''
        num_hidden: 중간 representation 크기
        num_latent: z의 dimension 
        '''
        super().__init__(*args, **kwargs)
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat x, y
        encoder_input = t.cat([x,y], dim=-1)

        # project vector with dim 3 
        encoder_input = self.input_projection(encoder_input)

        # self attention
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))

        # mu, sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        std = t.exp(0.5 * log_sigma)
        eps = t.rand_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_sigma, z 

class DeterministicEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim = 3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attention = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attention = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(2, num_hidden)
        self.target_projection = Linear(2, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = t.cat([context_x, context_y], dim=-1)   # [B, N, dim_x+y]

        encoder_input = self.input_projection(encoder_input)    # [B, N, num_hidden]

        for attention in self.self_attention:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for attention in self.cross_attention:
            query, _ = attention(keys, encoder_input, query)

        return query     

class Decoder(nn.Module):
    def __init__(self, num_hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_projection = Linear(2, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden*3, num_hidden*3, w_init='relu') for _ in range(3)])
        self.final_projection = Linear(num_hidden*3,1)

    def forward(self, target_x, r, z):
        batch_size, num_targets, _ = target_x.size() 
        target_x = self.target_projection(target_x)
        hidden = t.cat([t.cat([r,z], dim=-1), target_x], dim=-1)
        for linear in self.linears:
            hidden = linear(hidden)
        y_pred = self.final_projection(hidden)

        return y_pred

class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)
    def forward(self, key, value, query):
        # bmm : batch matrix multiplication 
        attn = t.bmm(query, key.transpose(1,2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = t.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        result = t.bmm(attn, value)

        return result, attn 

class Attention(nn.Module):
    def __init__(self, num_hidden, h=4,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h 
        self.h = h 

        self.key = Linear(num_hidden, num_hidden, bias=False) 
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        
        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)
        
        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


