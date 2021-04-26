import torch
from transformers import BertTokenizer
from models.bert import Model
from config import device, root_path, is_cuda, bert_path, max_length

class Predict(object):
    def __init__(self,
                 model_path=root_path + '/model/bert.ckpt',
                 bert_path=bert_path,
                 is_cuda=is_cuda):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.is_cuda = is_cuda
        bert_path = bert_path
        hidden_size = 768
        self.model = Model().to(device)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def process_data(self, text, is_cuda=is_cuda):
        def padding(indice, max_length, pad_idx=0):
            """
            pad 函数
            注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
            """
            pad_indice = [
                item + [pad_idx] * max(0, max_length - len(item))
                for item in indice
            ]
            return torch.tensor(pad_indice)

        text_dict = self.tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            ad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            #                                                    return_tensors='pt',     # Return pytorch tensors.
        )

        input_ids, attention_mask, token_type_ids = text_dict[
            'input_ids'], text_dict['attention_mask'], text_dict[
                'token_type_ids']

        token_ids_padded = padding([input_ids], max_length)
        token_type_ids_padded = padding([token_type_ids], max_length)
        attention_mask_padded = padding([attention_mask], max_length)
        return token_ids_padded, token_type_ids_padded, attention_mask_padded

    def predict(self, text):
        token_ids_padded, token_type_ids_padded, attention_mask_padded = self.process_data(
            text)
        if self.is_cuda:
            token_ids_padded = token_ids_padded.to(torch.device('cuda'))
            token_type_ids_padded = token_type_ids_padded.to(
                torch.device('cuda'))
            attention_mask_padded = attention_mask_padded.to(
                torch.device('cuda'))
        outputs = self.model(
            (token_ids_padded, attention_mask_padded, token_type_ids_padded))
        label = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
        score = torch.nn.functional.softmax(outputs.data[0])[torch.max(
            outputs.data, 1)[1].cpu().numpy()[0]].cpu().numpy().tolist()
        return label, score