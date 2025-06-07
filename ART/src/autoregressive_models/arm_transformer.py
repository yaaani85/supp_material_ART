
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoregressive_models.arm import AutoRegressiveModel


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dropout_probability, num_emb: int, num_heads: int = 8) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_probability)
        # hyperparams
        self.D = num_emb
        self.H = num_heads
        # weights for self-attention
        # self.w_k = nn.Linear(self.O, self.O * self.H)
        self.w_k = nn.Linear(self.D, self.D * self.H)
        self.w_q = nn.Linear(self.D, self.D * self.H)
        self.w_v = nn.Linear(self.D, self.D * self.H)
        # weights for a combination of multiple heads
        self.w_c = nn.Linear(self.D * self.H, self.D)

    def forward(self, x: torch.Tensor, causal: bool = True):
        # x: B(atch) x T(okens) x D(imensionality)
        B, T, D = x.size()
        # keys, queries, values
        k = self.w_k(x).view(B, T, self.H, D)  # B x T x H x D
        q = self.w_q(x).view(B, T, self.H, D)  # B x T x H x D
        v = self.w_v(x).view(B, T, self.H, D)  # B x T x H x D
        # TODO REDUCE V
        k = k.transpose(1, 2).contiguous().view(
            B * self.H, T, D)  # B*H x T x D
        q = q.transpose(1, 2).contiguous().view(
            B * self.H, T, D)  # B*H x T x D
        v = v.transpose(1, 2).contiguous().view(
            B * self.H, T, D)  # B*H x T x D

        k = k / (D**0.25)  # scaling
        q = q / (D**0.25)  # scaling

        # kq
        kq = torch.bmm(q, k.transpose(1, 2))  # B*H x T x T

        if causal:
            mask = torch.triu_indices(T, T, offset=1)
            kq[..., mask[0], mask[1]] = float('-inf')
        skq = F.softmax(kq, dim=2)
        # self-attention
        sa = torch.bmm(skq, v)  # B*H x T x D
        sa = sa.view(B, self.H, T, D)  # B x H x T x D
        sa = sa.transpose(1, 2)  # B x T x H x D
        sa = sa.contiguous().view(B, T, D * self.H)  # B x T x D*H

        out = self.w_c(sa)  # B x T x D
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dropout_probability, emb_dim: int, num_heads: int, num_neurons: int) -> None:
        super().__init__()

        # hyperparams
        self.D = emb_dim
        self.H = num_heads
        self.neurons = num_neurons
        # components
        self.msha = MultiHeadSelfAttention(dropout_probability, num_emb=self.D, num_heads=self.H)
        self.layer_norm = nn.LayerNorm(self.D)
        self.layer_norm2 = nn.LayerNorm(self.D)

        self.mlp = nn.Sequential(nn.Linear(
            self.D, self.neurons * self.D), nn.GELU(), nn.Linear(self.neurons * self.D, self.D))

    def forward(self, x: torch.Tensor, causal: bool = True):
        # Multi-Head Self-Attention
        x_attn = self.msha(x, causal)
        # LayerNorm
        x_normalized = self.layer_norm(x_attn + x)

        x_mlp = self.mlp(x_normalized)

        x_mlp_normalized = self.layer_norm2(x_mlp + x)

        return x_mlp_normalized


# TODO it seems causal is on every method, is it a property of the block isntead?
class ARMTransformer(AutoRegressiveModel):
    def __init__(self, number_of_blocks: int, emb_dim: int, dropout_probability: float, num_heads: int, num_neurons: int, n_entities: int, n_relations: int) -> None:
        # TODO: args seems liike a lazy thing, why not let the user unpack the argments?
        super().__init__()

        print('Transformer inspired by JT.')
        self.num_blocks: int = number_of_blocks
        self.n_entities = n_entities
        # positional embedding
        # self.positional_embedding = nn.Embedding(number_of_tokens, emb_dim)

        # transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(number_of_blocks):
            self.transformer_blocks.append(
                TransformerBlock(dropout_probability, emb_dim, num_heads, num_neurons))

        self.logits_relations = nn.Sequential(
            nn.Linear(emb_dim, n_relations),
            nn.Softmax(dim=-1)
        )

        self.logits_entities = nn.Sequential(
            nn.Linear(emb_dim, n_entities),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(dropout_probability)

    def transformer_forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # x: B X T
        e1_embedding, rel_embedding = x_tuple

        x = torch.cat((e1_embedding, rel_embedding), dim=1)
        # X: B X T X D
        # pos = torch.arange(0, x.shape[1], dtype=torch.long).to(self.device)
        # pos_emb = self.positional_embedding(pos).to(self.device)
        # dropout of embedding of inputs
        x = self.dropout(x)

        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)

        x_relations, x_entities = torch.chunk(x, 2, dim=1)
        # output logits
        logits_relations = self.logits_relations(x_relations.squeeze(1))
        logits_entities = self.logits_entities(x_entities.squeeze(1))
        return logits_relations, logits_entities

    def forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # get log-probabilities

        logits_relations, logits_entities = self.transformer_forward(x_tuple)
        return logits_relations, logits_entities

