from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        self.args = args
        self.model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512 #512 
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)
        
        #增加align
        self.linearText = nn.Linear(512, 512)
        self.linearImage = nn.Linear(768, 512)

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        #torch.Size([32, 77, 512])
        image_features = output['vision_model_output']['last_hidden_state']
        #torch.Size([32, 50, 768])

        text_feature = output['text_model_output']['pooler_output']
        #torch.Size([32, 512])
        image_feature = output['vision_model_output']['pooler_output']
        #torch.Size([32, 768])


        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature) 
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)

        
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)
        #outputs_text = torch.argmax(text_score, -1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            
        
            text = self.linearText(text_feature) 
            #torch.Size([32, 512])
            image = self.linearImage(image_feature)
            #torch.Size([32, 512])
        
            #SupMMConLoss
            #mmcLoss = self.mmc_2(text, image, None, None, labels)
            
            #UniSMMConLoss
            # 
            #mmcLoss = self.mmc_2(text, image, logits_text, logits_image, labels)
            
            #UnSupMMConLoss
            #mmcLoss = self.mmc_2(text, image, None, None, None)

            # loss = loss_fuse + loss_text + loss_image + 0.1*mmcLoss

            #original
            loss = loss_fuse + loss_text + loss_image 

            outputs = (loss,) + outputs
        return outputs
    
    

    def mmc_2(self, f0, f1, p0, p1, l):
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        
        if p0 is not None:
            p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)

            p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

        if l is None: 
            return self.UnSupMMConLoss(f0, f1)
        elif p0 is None:
            return self.SupMMConLoss(f0, f1, l)
        else:
            return self.UniSMMConLoss(f0, f1, p0, p1, l)

    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

        return mean_log_pos

    def SupMMConLoss(self, feature_a, feature_b, labels, temperature=0.07):
        # compute the mask matrix
        labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
        mask = torch.eq(labels, labels.T).float()

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        return mean_log_pos.mean()


    def UnSupMMConLoss(self, feature_a, feature_b, temperature=0.07):

        # compute the mask matrix
        mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(torch.device("cuda"))
        

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
  
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
  
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        mean_log_pos = mean_log_pos.mean()

        return mean_log_pos


