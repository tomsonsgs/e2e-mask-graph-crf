"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
def length2mask(length,maxlength):
    size=list(length.size())
    length=length.unsqueeze(-1).repeat(*([1]*len(size)+[maxlength])).long()
    ran=torch.arange(maxlength).cuda()
    ran=ran.expand_as(length)
    mask=ran<length
    return mask.float().cuda()
class graphCRFmodel(torch.nn.Module):
    def __init__(self,config):
        """Base RNN Encoder Class"""
        super(graphCRFmodel, self).__init__()
        config.K=2
        self.K=config.K
        self.config=config
        self.label_emb=torch.nn.Embedding(config.out_label+config.out_int,128)
        self.linear1=torch.nn.Linear(128,128,bias=False)
        self.w=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,config.out_label+config.out_int,config.out_label+config.out_int)) for _ in range(1)])
        self.b=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(config.out_label+config.out_int)) for _ in range(1)])
        for item in self.w:
         torch.nn.init.xavier_normal_(item) 
    def load_model(self, load_dir):
#        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
#        else:
#            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
     
    def forward(self,outlogits,outlogits1,length):#b,s+1,l1+l2;b,s+1,s+1;b,s+1,l1+l2;s+1,s+1;l1+l2,l1+l2,in_logits,adjs,tagmask,distance
        config=self.config
        pad1=torch.zeros(outlogits.size(0),outlogits.size(1),self.config.out_int).cuda().float()
        pad2=torch.zeros(outlogits.size(0),self.config.out_label).cuda().float()
        in_logits=torch.cat([outlogits,pad1],-1)
        in_logits1=torch.cat([pad2,outlogits1],-1)
        in_logits=torch.cat([in_logits,in_logits1.unsqueeze(-2)],-2)#b,s+1,l1+l2
        distance=torch.zeros(in_logits.size(1),in_logits.size(1)).cuda().float()
        for j in range(distance.size(0)):
            for jj in range(distance.size(1)):
              distance[j,jj]=abs(jj-j)
        for j in range(distance.size(0)-1):
            distance[j,-1]=1
        for j in range(distance.size(0)-1):
            distance[-1,j]=1
#        adjs=torch.zeros(in_logits.size(0),in_logits.size(1),in_logits.size(1)).cuda().float()#b,s+1,s+1
        adjs=torch.ones(in_logits.size(0),in_logits.size(1),in_logits.size(1)).cuda().float()#b,s+1,s+1
#        lenlist=length.cpu().data.tolist()
        lenlist=length
        for i in range(adjs.size(0)):
            adjs[i][:,lenlist[i]:-1]=0
#            for j in range(lenlist[i]):
#                if j>0:
#                    adjs[i][j,j-1]=1
#                if j<lenlist[i]-1:
#                    adjs[i][j,j+1]=1
#                adjs[i][j,-1]=1
#            for j in range(lenlist[i]):
#                adjs[i][-1,j]=1
            
            for j in range(adjs.size(1)):
                adjs[i][j,j]=0
        tagmask=torch.zeros(in_logits.size(0),in_logits.size(1),self.config.out_label+self.config.out_int).cuda().float()#b,s+1,l1+l2
        for i in range(tagmask.size(0)):
            tagmask[i][:-1,:self.config.out_label]=1
            tagmask[i][-1,self.config.out_label:]=1
            
            
            
        transfermask=torch.ones(config.out_label+config.out_int,config.out_label+config.out_int).float().cuda()
#        for j in range(config.out_label,config.out_label+config.out_int):
#            for i in range(config.out_label,config.out_label+config.out_int):
#                transfermask[j,i]=0
        for j in range(config.out_label+config.out_int):
            transfermask[j,j]=0
        transfer=torch.matmul(self.linear1(self.label_emb.weight),self.label_emb.weight.transpose(0,1))#l1+l2,l1+l2
        transfer=transfer+(1-transfermask)*-1e10
        transfer=torch.sigmoid(transfer)
        for i in range(transfer.size(0)):
           transfer.data[i,i]=1.0 
        transfer=transfer.unsqueeze(0).unsqueeze(0).repeat(in_logits.size(0),in_logits.size(1),1,1)

        in_logits=in_logits+(1-tagmask)*-1e10
        distance=torch.exp(-0.1*distance*distance)
        dis=torch.softmax(in_logits,-1)
        for j in range(self.K):
            dis1=torch.matmul(adjs*distance.unsqueeze(0),dis*tagmask)#b,s+1,l1+l2
#            dis1=dis1*tagmask
            dis1=torch.matmul(dis1,self.w[0].repeat(dis1.size(0),1,1))+self.b[0]#b,s+1,l1+l2
#            dis1=dis1.unsqueeze(-1)#b,s+1,l1+l2,1
#            dis1=torch.matmul(transfer,dis1).squeeze(-1)#b,s+1,l1+l2
            dis1=dis1+in_logits
            dis1=dis1+(1-tagmask)*-1e10
            if j< self.K-1:
               dis=torch.softmax(dis1,-1)
        outlogits2=dis1
        outlogits=outlogits2[:,:-1,:self.config.out_label]
        outlogits1=outlogits2[:,-1,self.config.out_label:]
        return outlogits,outlogits1
class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=True, num_layers=1, dropout=0., device=None, extFeats_dim=None, elmo_model=None, pretrained_model=None, pretrained_model_type=None, fix_pretrained_model=False):
        """Initialize model."""
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        #self.pad_token_idxs = pad_token_idxs
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.extFeats_dim = extFeats_dim

        self.num_directions = 2 if self.bidirectional else 1
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.elmo_model = elmo_model
        self.pretrained_model = pretrained_model
        self.pretrained_model_type = pretrained_model_type
        self.fix_pretrained_model = fix_pretrained_model
        if self.fix_pretrained_model:
            self.number_of_last_hiddens_of_pretrained = 4
            self.weighted_scores_of_last_hiddens = nn.Linear(self.number_of_last_hiddens_of_pretrained, 1, bias=False)
            for weight in self.pretrained_model.parameters():
                weight.requires_grad = False
        else:
            self.number_of_last_hiddens_of_pretrained = 1
        if self.elmo_model and self.pretrained_model:
            self.embedding_dim = self.elmo_model.get_output_dim() + self.pretrained_model.config.hidden_size
        elif self.elmo_model:
            self.embedding_dim = self.elmo_model.get_output_dim()
        elif self.pretrained_model:
            self.embedding_dim = self.pretrained_model.config.hidden_size
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.append_feature_dim = 0
        if self.extFeats_dim:
            self.append_feature_dim += self.extFeats_dim
            self.extFeats_linear = nn.Linear(self.append_feature_dim, self.append_feature_dim)
        else:
            self.extFeats_linear = None

        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim + self.append_feature_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions * self.hidden_dim, self.tagset_size)

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        if not self.elmo_model and not self.pretrained_model:
            self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        #for pad_token_idx in self.pad_token_idxs:
        #    self.word_embeddings.weight.data[pad_token_idx].zero_()
        if self.fix_pretrained_model:
            self.weighted_scores_of_last_hiddens.weight.data.uniform_(-initrange, initrange)
        if self.extFeats_linear:
            self.extFeats_linear.weight.data.uniform_(-initrange, initrange)
            self.extFeats_linear.bias.data.uniform_(-initrange, initrange)
        for weight in self.lstm.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)
    
    def forward(self, sentences, lengths, extFeats=None, with_snt_classifier=False, masked_output=None):
        # step 1: word embedding
        if self.elmo_model and self.pretrained_model:
            elmo_embeds = self.elmo_model(sentences['elmo'])
            elmo_embeds = elmo_embeds['elmo_representations'][0]
            tokens, segments, selects, copies, attention_mask = sentences['transformer']['tokens'], sentences['transformer']['segments'], sentences['transformer']['selects'], sentences['transformer']['copies'], sentences['transformer']['mask']
            outputs = self.pretrained_model(tokens, token_type_ids=segments, attention_mask=attention_mask)
            if self.fix_pretrained_model:
                pretrained_all_hiddens = outputs[2]
                used_hiddens = torch.cat([hiddens.unsqueeze(3) for hiddens in pretrained_all_hiddens[- self.number_of_last_hiddens_of_pretrained:]], dim=-1)
                pretrained_top_hiddens = self.weighted_scores_of_last_hiddens(used_hiddens).squeeze(3)
            else:
                pretrained_top_hiddens = outputs[0]
            batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
            chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            pretrained_embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            pretrained_embeds = pretrained_embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
            embeds = torch.cat((elmo_embeds, pretrained_embeds), dim=2)
        elif self.elmo_model:
            elmo_embeds = self.elmo_model(sentences)
            embeds = elmo_embeds['elmo_representations'][0]
        elif self.pretrained_model:
            tokens, segments, selects, copies, attention_mask = sentences['tokens'], sentences['segments'], sentences['selects'], sentences['copies'], sentences['mask']
            outputs = self.pretrained_model(tokens, token_type_ids=segments, attention_mask=attention_mask)
            if self.fix_pretrained_model:
                pretrained_all_hiddens = outputs[2]
                used_hiddens = torch.cat([hiddens.unsqueeze(3) for hiddens in pretrained_all_hiddens[- self.number_of_last_hiddens_of_pretrained:]], dim=-1)
                pretrained_top_hiddens = self.weighted_scores_of_last_hiddens(used_hiddens).squeeze(3)
            else:
                pretrained_top_hiddens = outputs[0]
            batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
            chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        else:
            embeds = self.word_embeddings(sentences)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        concat_input = self.dropout_layer(concat_input)
        
        # step 2: BLSTM encoder
        packed_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_lstm_out, packed_h_t_c_t = self.lstm(packed_embeds)  # bsize x seqlen x dim
        lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_lstm_out, batch_first=True)

        # step 3: slot tagger
        lstm_out_reshape = lstm_out.contiguous().view(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(lstm_out_reshape))
        tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = tag_scores.view(lstm_out.size(0), lstm_out.size(1), tag_space.size(1))
        tag_space=tag_space.view(lstm_out.size(0), lstm_out.size(1), tag_space.size(1))
        if with_snt_classifier:
            return tag_scores,tag_space,(packed_h_t_c_t, lstm_out, lengths)
        else:
            return tag_scores
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

