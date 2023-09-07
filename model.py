import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.embed = 768
        self.hidden_size = 768
        self.gru = nn.GRU(self.embed, self.hidden_size, batch_first=True)
        self.ws1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ws2 = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    # 输入：[batch, sen_len, embed_dim] （batch， 句子长度， 词向量维度）
    # 输出：[batch, hidden_size] (batch， 隐藏层维度)
    def forward(self, vector):
        embed, _ = self.gru(vector)  # [batch, sen_len, embed_dim]
        atten_out = self.attention(vector)  # [batch, hidden_size]
        return atten_out

    def attention(self, embedding):
        self_attention = torch.tanh(self.ws1(embedding))  # [sen_num,sen_len,2*hidden_dim]
        self_attention = self.ws2(self_attention).squeeze(2)  # [sen_num,sen_len]
        self_attention = self.softmax(self_attention)
        sent_encoding_ = torch.sum(embedding * self_attention.unsqueeze(-1), dim=1)
        return sent_encoding_


if __name__ == '__main__':
    att = Attention()
    vec = torch.randn(5, 32, 768)
    out = att(vec)
