import torch.nn as nn

class BERT_Spam(nn.Module):
    def __init__(self, bert):
      super(BERT_Spam, self).__init__()
      self.bert = bert

      # dropout layer
      self.dropout = nn.Dropout(0.5)

      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,256)

      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(256,2)

      #softmax activation function
      #self.softmax = nn.LogSoftmax(dim=1)
      self.softmax = nn.Softmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model
      #print('before output')
      output = self.bert(sent_id, attention_mask=mask)
      #print(output)
      #hidden_state = output[1]
      #print(hidden_state.shape)

      # For handling distilbert output
      hidden_state = output[0][:, 0, :]
      #print(hidden_state.shape)
      #print('after output')
      x = self.fc1(hidden_state)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      # apply softmax activation
      #x = self.softmax(x)

      return x