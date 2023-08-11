

# Models for ViT Cycle GAN


# Design by HanLin


import warnings
warnings.filterwarnings('ignore')


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat


# Discriminator test 0 START =======


# In order to save resources, first try CNN model
class Discriminator(nn.Module):
    def __init__(self,input_channel):
        super(Discriminator,self).__init__()

        model = []

        model.extend([
            nn.Conv2d(input_channel,64,3,stride=2),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(64,128,3,stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(128,256,3,stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(256,512,3,stride=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        self.model = nn.Sequential(*model)

        self.adamaxpool = nn.AdaptiveMaxPool2d(1)

        self.linear = nn.Linear(512,1)

    def forward(self,x):
        x = self.model(x)
        x = self.adamaxpool(x)

        x = x.view(x.size(0),-1)

        x = self.linear(x)

        return x


# Discriminator test 0 END =======


# Generator test 0 START =======


class Encoder_Config():
    def __init__(self,
            patch_size,
            in_channels,
            out_channels,
            sample_rate=4,
            hidden_size=1024,
            num_hidden_layers=8,
            num_attention_heads=6,
            intermediate_size=1024,

            input_dense_drop_rate=0,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            ):

        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.input_dense_drop_rate=input_dense_drop_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


# Gaussian Error Linear Units (GELUS)
def Gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Layer_Norm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(Layer_Norm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Self_Attention(nn.Module):
    def __init__(self,config:Encoder_Config):
        super().__init__()

        assert config.hidden_size%config.num_attention_heads == 0,r'Self_Attention : hidden size is not a multiple of the number of attention !'

        # Actually useless
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = [x.size()[0],x.size()[1],self.num_attention_heads,self.attention_head_size]
        x = x.view(*new_x_shape)
        # (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self,hidden_states):
        # q=Wa
        mixed_query_layer = self.query(hidden_states)
        # k=Wa
        mixed_key_layer = self.key(hidden_states)
        # v=Wa
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention=q@k
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores/math.sqrt(self.attention_head_size)

        # attention=Softmax(attention)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.dropout(attention_probs)

        # out=attention@v
        context_layer = torch.matmul(attention_probs, value_layer)
        # [batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = [context_layer.size()[0],context_layer.size()[1],self.all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class Single_Output(nn.Module):
    def __init__(self, config,input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, config.hidden_size)
        self.LayerNorm = Layer_Norm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Half of the FFN after Attention
class Middle_Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = Gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Single encoder
class Single_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = Self_Attention(config)
        self.attention_output = Single_Output(config, config.hidden_size)

        self.middle_output = Middle_Output(config)
        self.output = Single_Output(config, config.intermediate_size)

    def forward(self,hidden_states):
        self_outputs = self.self_attention(hidden_states)

        print('hidden_states shape : ',hidden_states.shape)
        print('self_outputs shape : ', self_outputs.shape)

        attention_output = self.attention_output(self_outputs, hidden_states)

        print('attention_outputs shape : ',attention_output.shape)

        # Skip connection
        attention_output=hidden_states+attention_output

        # FFN 0 after the Attention
        intermediate_output = self.middle_output(attention_output)
        # FFN 1 after the Attention
        layer_output = self.output(intermediate_output, attention_output)

        print('intermediate_output shape : ',intermediate_output.shape)
        print('layer_output shape : ',layer_output.shape)

        # Skip connection
        layer_output=attention_output+layer_output

        return layer_output


# Embedding (a=Wx)
class InputDense(nn.Module):
    def __init__(self, config):
        super(InputDense, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels, config.hidden_size)
        self.transform_act_fn = Gelu
        self.LayerNorm = Layer_Norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Convolution Embedding (a=Wx)
class InputDense_conv(nn.Module):
    def __init__(self,config,out_channels=64, kernel_size=3, stride=1,act_layer=nn.LeakyReLU):
        super(InputDense_conv, self).__init__()

        self.dense=nn.Conv2d(config.in_channels,out_channels,kernel_size=3,stride=stride,padding=kernel_size//2)
        self.transform_act_fn=act_layer(inplace=True)
        self.pos_drop=nn.Dropout(p=config.input_dense_drop_rate)

        config.in_channels=out_channels

    def forward(self,x):
        x=self.dense(x)
        x=self.transform_act_fn(x)
        x=self.pos_drop(x)

        return x


# Positional embedding (a=a+p)
class Positional_Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = Layer_Norm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        input_shape = input_ids.size()

        seq_length = input_shape[1]
        device = input_ids.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransModel(nn.Module):

    def __init__(self, config):
        super(TransModel, self).__init__()

        # self.inputdense = InputDense(config)
        # self.embeddings = Positional_Embeddings(config)

        encoder_layers = [Single_encoder(config) for _ in range(config.num_hidden_layers)]
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self,input_ids):
        # dense_out = self.inputdense(input_ids)
        # embedding_output = self.embeddings(dense_out)

        embedding_output=input_ids
        encoder_layers = self.encoder_layers(embedding_output)

        return encoder_layers


class Encoder(nn.Module):
    def __init__(self, config:Encoder_Config):
        super().__init__()

        self.config = config

        self.conv_inputdense=InputDense_conv(config)

        self.bert_model = TransModel(config)

        sample_v = int(math.pow(2,config.sample_rate))

        assert config.patch_size[0]*config.patch_size[1]*config.hidden_size%(sample_v**2) == 0,r'Encoder : No divisible !'

        self.final_dense = nn.Linear(config.hidden_size,config.patch_size[0]*config.patch_size[1]*config.hidden_size//(sample_v**2))

        self.hh = config.patch_size[0] // sample_v
        self.ww = config.patch_size[1] // sample_v

    def forward(self, x):

        x=self.conv_inputdense(x)

        # x : (b, c, w, h)
        b, c, h, w = x.shape

        p1 = self.config.patch_size[0]
        p2 = self.config.patch_size[1]

        if (h % p1 != 0) or (w % p2 != 0):
            print(r'Encoder : Image size, no divisible !')
            os._exit(0)

        hh = h // p1
        ww = w // p2

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)

        x = self.bert_model(x)

        x = self.final_dense(x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=self.hh, p2=self.ww, h=hh, w=ww,c=self.config.hidden_size)

        return x


class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,features=[512, 256, 128, 64]):
        super().__init__()

        self.decoder_0 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_0(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.final_out(x)
        return x


class Generator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 patch_size=(32, 32),
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 decode_features=[512, 256, 128, 64],
                 sample_rate=4
                 ):
        super().__init__()

        config = Encoder_Config(
                             patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             sample_rate=sample_rate,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads
                             )

        self.encoder = Encoder(config)
        self.decoder = Decoder(in_channels=config.hidden_size, out_channels=config.out_channels,features=decode_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Generator test 0 END =======


















