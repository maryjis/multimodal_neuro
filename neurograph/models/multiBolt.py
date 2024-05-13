import numpy as np
import torch
from torch import nn
from neurograph.models.bolT import BolT
from neurograph.config.config import MultiModalBoltConfig, MultiModalMorphBoltConfig, MultiModalMorphBoltV2Config
from torch.nn import functional as F

class MultiModalBolt(BolT):
    
    def __init__(self,         
                input_dim_1: int,
                input_dim_2: int,
                num_nodes_1: int,  # used for concat pooling
                num_nodes_2: int,
                model_cfg: MultiModalBoltConfig):
        
            assert num_nodes_1==num_nodes_2, "ROIs Number must be equal"
            super().__init__(input_dim=input_dim_1, 
                         num_nodes=num_nodes_1,
                         model_cfg =model_cfg)
            self.weighted_m =nn.Linear(num_nodes_1, num_nodes_1)
            self.comb =nn.Linear(num_nodes_1, 1, bias=False)
    
    
    def forward(self, batch, analysis=False):
        
        x_fmri, x_dti, y = batch
        
        x_fmri =  torch.unsqueeze(x_fmri.permute((0, 2, 1)), 2)
        
        batch_size, _, n_roi =x_dti.shape
        
        x_dti =self.weighted_m(x_dti.reshape(-1,n_roi)).reshape(batch_size,n_roi,n_roi)
        
        x_dti =torch.unsqueeze(x_dti, 1)


        x_comb = torch.mul(x_fmri,x_dti)
        
        x_comb =self.comb(x_comb).squeeze().permute((0, 2, 1))
        
        return super().forward((x_comb, y), analysis)


class MultiModalMorphBoltV2(BolT):
    
    def __init__(self,         
                input_dim: int,
                num_nodes: int,
                morph_input_dim: int,
                model_cfg: MultiModalMorphBoltV2Config):
        
        super().__init__(input_dim=input_dim, 
                         num_nodes=num_nodes,
                         model_cfg =model_cfg)

        self.model_type = model_cfg.model_type

        self.cross_modality_possition = model_cfg.cross_modality_possition
        self.mmtr = nn.Linear(morph_input_dim, model_cfg.dim)
    
    
    def forward(self, batch, analysis=False):
        
        roiSignals, morph, y = batch

        batch_size, R, T = roiSignals.shape
        _, M = morph.shape

        if self.model_type == 'original':
            roiSignals_comb = roiSignals
        elif self.model_type == 'last':
            roiSignals_comb = torch.empty(batch_size, R, T + 1).to('cuda:0')
            morph_transformed = self.mmtr(morph)
            for i in range(batch_size):
                cur_roiSignal = roiSignals[i]
                cur_morph = morph_transformed[i].view(-1, 1)
                cur_roiSignal = torch.cat((cur_roiSignal, cur_morph), axis=1)
                roiSignals_comb[i] = cur_roiSignal
        elif self.model_type == 'cross_modality':
            roiSignals_comb = torch.empty(batch_size, R, T + T // self.cross_modality_possition).to('cuda:0')
            morph_transformed = self.mmtr(morph)
            for i in range(batch_size):
                cur_roiSignal = roiSignals[i]
                cur_morph = morph_transformed[i].view(-1, 1)
                insert_indices = range(self.cross_modality_possition, T, self.cross_modality_possition)
                for index in insert_indices:
                    cur_roiSignal = torch.cat((cur_roiSignal[:, :index], 
                                        cur_morph, cur_roiSignal[:, index:]), dim=1)
                roiSignals_comb[i] = cur_roiSignal

        return super().forward((roiSignals_comb, y), analysis)


class MultiModalMorphBolt(BolT): 
    
    def __init__(self,         
                input_dim: int,
                num_nodes: int,
                morph_input_dim: int,
                model_cfg: MultiModalMorphBoltConfig):
        
        super().__init__(input_dim=input_dim, 
                         num_nodes=num_nodes,
                         model_cfg =model_cfg)
        
        self.fusion_type = model_cfg.fusion_type
        if self.fusion_type =="sum" or self.fusion_type =="late":
            self.lin_morph =nn.Sequential(
                nn.Linear(morph_input_dim, model_cfg.n_classes),
                nn.BatchNorm1d(model_cfg.n_classes),
                nn.Dropout(p=model_cfg.fusion_dropout))
                
        elif self.fusion_type == "concat":     
            self.lin_morph =nn.Sequential(
                nn.Linear(morph_input_dim, model_cfg.fusion_dim),
                nn.ReLU(),
                nn.BatchNorm1d(model_cfg.fusion_dim),
                nn.Dropout(p=model_cfg.fusion_dropout))
            self.layer_norm = nn.LayerNorm(model_cfg.dim+ model_cfg.fusion_dim)
            self.classifierHead =nn.Linear(model_cfg.dim+ model_cfg.fusion_dim, model_cfg.n_classes)
        else:
            raise NotImplementedError 
        
    def forward(self, batch, analysis=False):
        """
        Input :

            roiSignals : (batchSize, N, dynamicLength)

            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise

        Output:

            logits : (batchSize, #ofClasses)


        """
        roiSignals, morph, y = batch
        
        # normalize morpometric vector 
        morph =(morph -torch.mean(morph, axis =0)) / (1+torch.std(morph, axis =0))
        
        morph_representation =self.lin_morph(morph)
        
        roiSignals = roiSignals.permute((0, 2, 1))

        batchSize = roiSignals.shape[0]
        T = roiSignals.shape[1]  # dynamicLength

        nW = (T - self.model_cfg.windowSize) // self.shiftSize + 1
        # (batchSize, #windows, C)
        cls = self.clsToken.repeat(batchSize, nW, 1)

        # record nW and dynamicLength, need in case you want to paint those tokens later
        self.last_numberOfWindows = nW

        if analysis:  # pragma: no cover
            self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        for block in self.blocks:
            roiSignals, cls = block(roiSignals, cls, analysis)

            if analysis:  # pragma: no cover
                self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        """
            roiSignals : (batchSize, dynamicLength, featureDim)
            cls : (batchSize, nW, featureDim)
        """

        cls = self.encoder_postNorm(cls)

        if self.model_cfg.pooling == "cls":
            cls =cls.mean(dim=1)
        elif self.model_cfg.pooling == "gmp":  # pragma: no cover
            cls = roiSignals.mean(dim=1)
        
        # concatenate multimodal representations and pass them into linear layer to dim reduction = hidden_dim 
        
        if self.fusion_type == "concat":
            cls_concat = torch.concat([cls, morph_representation], axis=1)
            cls_concat =self.layer_norm(cls_concat)
            logits = self.classifierHead(cls_concat)
        else:        
            logits = self.classifierHead(cls)  # (batchSize, #ofClasses)
        
        if self.fusion_type == "sum": 
            logits = morph_representation +logits
        
        if self.fusion_type =="late":
            return logits, morph_representation, cls
        
        torch.cuda.empty_cache()

        return logits, cls    
        
               
        
        
        
    