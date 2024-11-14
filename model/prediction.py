import torch
import torch.nn as nn


'''
multimodal_concat 추가 
'''

class TextAudioClassifier(nn.Module):
    # def __init__(self, audio_model, text_model, audio_dim, text_dim = 768, hidden_dim, num_classes=6, dropout_prob=0.5):
    def __init__(self, audio_model = None, speaker_model = None, text_model = None, 
                 audio_dim=512, text_dim=None, hidden_dim=256, num_classes=4, dropout_prob=0.5, 
                 speaker='None', speaker_dim=10, cross_modal_atten = False, modal = 'audio'):

        super(TextAudioClassifier, self).__init__()
        self.speaker = speaker
        
        self.audio_model = audio_model
        self.speaker_model = speaker_model
        self.text_model = text_model
        
        self.audio_dim, self.text_dim, self.hidden_dim, self.speaker_dim = audio_dim, text_dim, hidden_dim, speaker_dim
        self.num_classes = num_classes
        self.modal = modal 
        self.cross_modal_atten = cross_modal_atten

        self.dropout_prob = dropout_prob
        
        if self.audio_model is not None: 
            # [Audio + Text Multimodal]
            if self.text_model is not None:
                if cross_modal_atten: 
                    self.input_dim = text_dim 
                    if modal in ['multimodal_concat']:
                        self.input_dim = text_dim + audio_dim        
                else: self.input_dim = text_dim + audio_dim

            
            # [Audio Unimodal] 
            else: self.input_dim = audio_dim 
        
        # [Text Unimodal] 
        else: self.input_dim  = text_dim

        # [Speaker Emb]   
        if self.modal in ['audio', 'multimodal', 'multimodal_concat']:            
            if self.speaker in ["speaker_emb", "speaker_emb_0"]:
                self.input_dim  += audio_dim 
                self.speaker_emb = nn.Embedding(self.speaker_dim, audio_dim)
                if self.speaker == "speaker_emb_0":
                    nn.init.constant_(self.speaker_emb.weight, 0.0)
                    self.speaker_emb.weight.requires_grad = False
            elif self.speaker == "wavlm": 
                self.input_dim  += self.speaker_dim 
                self.speaker_emb = self.speaker_model
        else:
            if self.speaker in ["speaker_emb", "speaker_emb_0"]:
                self.input_dim  += text_dim 
                self.speaker_emb = nn.Embedding(self.speaker_dim, text_dim)
                if self.speaker == "speaker_emb_0":
                    nn.init.constant_(self.speaker_emb.weight, 0.0)
                    self.speaker_emb.weight.requires_grad = False
            
        self.pred_linear = nn.Sequential(
            nn.Linear(self.input_dim , self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        self.initialize_weights()
        
    def forward(self, audio_input, text_input = None, speaker_ID = None,  length=None):
        
        # Unimodal [AUDIO]
        if self.modal =='audio': feature = self.audio_model(audio_input, length)

        # Unimodal [TEXT]
        elif self.modal=='text': feature = self.text_model(embeddings=text_input)

        # Multimodal [AUDIO + TEXT]
        elif self.modal in ['multimodal', 'multimodal_concat']:
            if self.cross_modal_atten: # cross modal attention 
                if self.modal in ['multimodal']: 
                    audio_feature, mask = self.audio_model(audio_input, length)
                    # print("self.text_model", self.text_model)
                    feature = self.text_model(embeddings = text_input, acoustic_encode = audio_feature, a_attention_mask=mask)
                    
                elif self.modal in ['multimodal_concat']: # cross modal attention + concat audio feature 
                    # print("self.text_model", self.text_model)
                    audio_feature1, mask, audio_feature2 = self.audio_model(audio_input, length)
                    feature = self.text_model(embeddings = text_input, acoustic_encode = audio_feature1, a_attention_mask=mask)
                    
                    feature = torch.cat((feature, audio_feature2), dim=-1)
            else: 
                 # concat multimodal feature 
                 feature = self.audio_model(audio_input, length)
                 txt_feature = self.text_model(embeddings=text_input)
                #  print(f"audio feature: {feature.shape}, txt_feature: {txt_feature.shape}")
                 feature = torch.cat((feature, txt_feature), dim=-1)

        # if self.modal in ['audio', 'multimodal']:
        if self.speaker == "wavlm":
            speaker_emb = self.speaker_emb(audio_input, length)
            feature = torch.cat((feature, speaker_emb), dim=-1) 

        elif self.speaker in ["speaker_emb", "speaker_emb_0"]: 
            speaker_emb = self.speaker_emb(speaker_ID)
            #print("speaker_emb: ",speaker_emb)
            feature = torch.cat((feature, speaker_emb), dim=-1)  
       
        output = self.pred_linear(feature)

        return output
    
    def initialize_weights(self):
        for m in self.pred_linear:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    