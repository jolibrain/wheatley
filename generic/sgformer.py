import dgl
import torch


def full_attention_conv(qs, ks, vs):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    return attn_output


class TransConvLayer(torch.nn.Module):
    """
    transformer with fast attention
    """

    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = torch.nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = torch.nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = torch.nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(
        self,
        query_input,
        source_input,
    ):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        return final_output


class TransConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_heads=1,
        use_weight=True,
    ):
        super().__init__()

        self.fc = torch.nn.Linear(in_channels, hidden_channels)
        self.conv = TransConvLayer(
            hidden_channels,
            hidden_channels,
            num_heads=num_heads,
            use_weight=use_weight,
        )

        self.activation = torch.nn.functional.relu

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x):
        # input MLP layer
        x = self.fc(x)
        x = self.activation(x)

        out = []
        for i in range(x.shape[0]):
            out.append(self.conv(x[i], x[i]))
        return torch.stack(out)
