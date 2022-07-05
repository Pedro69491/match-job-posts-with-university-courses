import numpy as np
import torch
from torch import nn
import transformers
from transformers import BertModel, BertTokenizer
from BERT import train_loader, valid_loader, test_loader, empty_arr, binary_classification, metrics



class BertLayer(nn.Module):
    def __init__(self, bert):
        super(BertLayer, self).__init__()

        self.bert = bert
        self.fc = nn.linear(768, 58)
    

    def forward(self, input, attention):

        out = self.bert(input, attention)

        x = self.dropout(self.fc(out))

        return x



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model = BertModel.from_pretrained('bert-base-uncased')
model = BertLayer(model)
optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

model.train()
for batch in train_loader:
    optim.zero_grad()
    
    inputs = batch['input_ids']
    attention = batch['attention_mask']
    labels = batch['labels'].to(torch.float16)
    logits = model(inputs, attention).logits
    
    loss = binary_classification(logits, labels)
    
    print(loss.items())
    
    loss.backward()
    optim.step()


a, r, f, p = empty_arr()
model.eval()
for i in test_loader:
    inputs = batch['input_ids']
    attention = batch['attention_mask']
    labels = batch['labels'].to(torch.float16).numpy()
    preds = torch.round(torch.sigmoid(model(inputs, attention).logits)).detach().numpy()

    a, r, f, p = metrics(labels, preds, a, r, f, p)
    
print('\nacc: {}\nrecall: {}\nf1: {}\nprecision: {}'.format(np.mean(a) 
                                                ,np.mean(r), np.mean(f), np.mean(p)))

