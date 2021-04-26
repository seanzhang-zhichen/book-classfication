# coding: UTF-8
import torch.nn as nn
from transformers import BertModel,BertConfig
from config import bert_path, num_classes, hidden_size

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model_config =BertConfig.from_pretrained(bert_path, num_labels=num_classes)
        self.bert = BertModel.from_pretrained(bert_path, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc =nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        output = self.bert(context,
                           attention_mask=mask,
                           token_type_ids=token_type_ids)

        # print('last hidden state shape')
        # print(output[0].shape)
        # print('pooler  shape')
        # print(output[1].shape)
        out = self.fc(output[1])
        return out
