import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_head=8, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.model_dim = model_dim

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

        self.d_k = self.d_v = model_dim // num_head  # 512 / 8 = 64

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def self_attention(self, q, k, v, mask=None):
        """

        Args:
            q:     query [B, token, 512]
            k:     key   [B, token, 512]
            v:     value [B, token, 512]
            mask:  mask  [B, 1, 1, 512]

        Returns: salced dot product results [B, 100, 512]

        """
        # 1. split as head      [b, TOKEN 512], -> [b, token, HEAD, 64] -> [B, num_head, token, 64]
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)

        # 2. q * k^t (transpose of k) & scaled dot product
        qk = torch.matmul(q, k.permute(0, 1, 3, 2))                                   # [B, num_head, token, token]
        attention_score = qk / math.sqrt(self.d_k)

        # softmax 앞에서 masking (정보가 없는 부분의 score 를 매우 작은 값으로 만듦)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e20)

        softmax_attention_score = torch.softmax(attention_score, dim=-1)              # [B, num_head, token, token]
        result = torch.matmul(softmax_attention_score, v)                             # [B, num_head, token, dk]

        # concat the multi-head result semantically
        result = result.permute(0, 2, 1, 3).reshape(batch_size, -1, self.model_dim)   # [B, token, h, dk -> B, 100, 512]
        return result

    def forward(self, x1, x2, x3, mask=None):
        residual = x1
        # x shape is                                     [batch, num_token, model_dim] - [4, 100, 512]
        q = self.W_q(x1)    # [B, 100, 512]
        k = self.W_k(x2)    # [B, 100, 512]
        v = self.W_v(x3)    # [B, 100, 512]

        if mask is not None:
            mask = mask.unsqueeze(1)                    # [B, 1, 1, 100] for broadcasting.

        attention = self.self_attention(q, k, v, mask)  # [B, 100, 512]
        x = self.W_o(attention)                         # [B, 100, 512]

        # Dropout-Residual-LayerNorm -> post LayerNorm original version of transformer
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffd = nn.Sequential(nn.Linear(model_dim, model_dim * 4),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(model_dim * 4, model_dim))

    def forward(self, x):
        residual = x
        x = self.ffd(x)
        # Dropout-Residual-LayerNorm -> post LayerNorm original version of transformer
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_head, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(model_dim=model_dim, num_head=num_head, dropout=dropout)
        self.feed_forward = FeedForwardNet(model_dim=model_dim, dropout=dropout)

    def forward(self, x, mask):
        x = self.multi_head_attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_head, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([])

        for _ in range(self.num_layers):
            self.encoder_layers.append(EncoderBlock(model_dim, num_head, dropout))

    def forward(self, x, mask=None):
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, mask)
        return x


class PoolingMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_head, num_seed, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seed, model_dim))
        nn.init.xavier_uniform_(self.S)
        self.multi_head_attention = MultiHeadAttention(model_dim=model_dim, num_head=num_head, dropout=dropout)

    def forward(self, x):
        x = self.multi_head_attention(self.S.repeat(x.size(0), 1, 1), x, x)
        return x


class SPHTransformer(nn.Module):

    def __init__(self, model_dim, num_patches, num_head=12, num_layers=12, dropout=0.1, num_classes=10, input_dim=64, is_classify=True):
        super().__init__()

        self.model_dim = model_dim
        self.num_patches = num_patches
        # number of patches (N)

        self.patch_embedding_projection = nn.Linear(input_dim, self.model_dim)

        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches, self.model_dim))  # [1, N, D]
        torch.nn.init.normal_(self.position_embedding, std=.02)  # 확인해보기

        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_head=num_head, dropout=dropout)
        self.is_classify = is_classify
        if is_classify:
            # self.classifier = nn.Linear(self.num_patches * self.model_dim, num_classes)
            self.classifier = nn.Linear(self.num_patches, num_classes)

        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, seq):
        batch_size = seq.size(0)
        x = self.patch_embedding_projection(seq)  # [B, 20, 64]  ->  [B, 20, model_dim]
        x += self.position_embedding  # (=E_pos)
        x = self.dropout(x)
        x = self.encoder(x)
        if self.is_classify:
            x = torch.mean(x, dim=-1)   # [B, 25]
            # x = x.reshape([batch_size, self.model_dim * self.num_patches])
            x = self.classifier(x)
        return x



class SPHTransformer_Classic(nn.Module):

    def __init__(self, model_dim, num_patches, num_head=12, num_layers=12, dropout=0.1, num_classes=10, input_dim=64, is_classify=True):
        super().__init__()

        self.model_dim = model_dim
        self.num_patches = num_patches
        # number of patches (N)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))                              # [1, 1, D]
        self.patch_embedding_projection = nn.Linear(input_dim, self.model_dim)

        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches + 1, self.model_dim))  # [1, N + 1, D]
        torch.nn.init.normal_(self.position_embedding, std=.02)  # 확인해보기

        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_head=num_head, dropout=dropout)
        self.is_classify = is_classify
        if is_classify:
            self.classifier = nn.Linear(self.model_dim, num_classes)

        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, seq):
        batch_size = seq.size(0)

        x = self.patch_embedding_projection(seq)  # [B, 21, 64]  ->  [B, 21, model_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)          # expand to [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)                           # [B, 1 + N, D]
        x += self.position_embedding  # (=E_pos)
        x = self.dropout(x)
        x = self.encoder(x)
        if self.is_classify:
            x = x[:, 0]
            x = self.classifier(x)
        return x


if __name__ == '__main__':

    # params : 57418
    image = torch.randn([2, 1280, 4 * 3])  # mnist icosahedron
    vit = SPHTransformer(model_dim=24, num_patches=1280, num_classes=10, num_head=8, num_layers=7, input_dim=4)
    output = vit(image)
    print(output.size())

    # vit_classic = SPHTransformer_Classic(model_dim=24, num_patches=20, num_classes=10, num_head=8, num_layers=7, input_dim=64)
    # output = vit_classic(image)
    # print(output.size())

    # # params : 59946
    # image = torch.randn([2, 20, 64])  # mnist icosahedron
    # vit = SPHTransformer(model_dim=32, num_patches=20, num_classes=10, num_head=8, num_layers=4, input_dim=64)
    # output = vit(image)
    # print(output.size())

    # image = torch.randn([2, 20, 256 * 3])
    # vit = SPHTransformer(model_dim=64, num_patches=20, num_classes=10, num_head=8, num_layers=12, input_dim=256 * 3)
    # output = vit(image)
    # print(output.size())
    #
    # image = torch.randn([2, 20, 256 * 3])
    # vit = SPHTransformer(model_dim=64, num_patches=20, num_classes=10, num_head=8, num_layers=12, input_dim=256 * 3)
    # output = vit(image)
    # print(output.size())
    #
    # image = torch.randn([2, 6, 225])
    # vit = SPHTransformer(model_dim=24, num_patches=6, num_classes=10, num_head=8, num_layers=6, input_dim=225)
    # output = vit(image)
    # print(output.size())
    #
    # image = torch.randn([2, 6, 841 * 3])
    # vit = SPHTransformer(model_dim=64, num_patches=6, num_classes=10, num_head=8, num_layers=12, input_dim=841 * 3)
    # output = vit(image)
    # print(output.size())