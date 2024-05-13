"""
Adapted from https://github.com/bolt/bolt

Copyright (c) 2013-2019 Bolt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from torch import nn

import numpy as np
import math
from neurograph.models.bolt.bolTransformerBlock import BolTransformerBlock

from neurograph.config.config import BolTConfig


class BolT(nn.Module):
    def __init__(self, input_dim: int, num_nodes: int, model_cfg: BolTConfig):

        super().__init__()

        dim = model_cfg.dim
        nOfClasses = model_cfg.n_classes

        self.model_cfg = model_cfg

        self.inputNorm = nn.LayerNorm(dim)

        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = []

        shiftSize = int(model_cfg.windowSize * model_cfg.shiftCoeff)
        self.shiftSize = shiftSize
        self.receptiveSizes = []

        for i, layer in enumerate(range(model_cfg.nOfLayers)):

            if model_cfg.focalRule == "expand":
                receptiveSize = model_cfg.windowSize + math.ceil(
                    model_cfg.windowSize
                    * 2
                    * i
                    * model_cfg.fringeCoeff
                    * (1 - model_cfg.shiftCoeff)
                )
            elif model_cfg.focalRule == "fixed":  # pragma: no cover
                receptiveSize = model_cfg.windowSize + math.ceil(
                    model_cfg.windowSize
                    * 2
                    * 1
                    * model_cfg.fringeCoeff
                    * (1 - model_cfg.shiftCoeff)
                )

            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(
                BolTransformerBlock(
                    dim=model_cfg.dim,
                    numHeads=model_cfg.numHeads,
                    headDim=model_cfg.headDim,
                    windowSize=model_cfg.windowSize,
                    receptiveSize=receptiveSize,
                    shiftSize=shiftSize,
                    mlpRatio=model_cfg.mlpRatio,
                    attentionBias=model_cfg.attentionBias,
                    drop=model_cfg.drop,
                    attnDrop=model_cfg.attnDrop,
                )
            )

        self.blocks = nn.ModuleList(self.blocks)

        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(dim, nOfClasses)

        # for token painting
        self.last_numberOfWindows = None

        # for analysis only
        self.tokens = []

        self.initializeWeights()
        
        if model_cfg.checkpoint:
            
            bolt_loaded_model =torch.load(model_cfg.checkpoint)
            print(bolt_loaded_model)
            
            self.inputNorm = bolt_loaded_model.inputNorm
            self.clsToken = bolt_loaded_model.clsToken
            self.blocks = bolt_loaded_model.blocks 
            self.encoder_postNorm = bolt_loaded_model.encoder_postNorm  

    def initializeWeights(self):
        # a bit arbitrary
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def calculateFlops(self, T):  # pragma: no cover

        windowSize = self.model_cfg.windowSize
        shiftSize = self.shiftSize
        focalSizes = self.focalSizes

        macs = []

        nW = (T - windowSize) // shiftSize + 1

        C = 400  # for schaefer atlas
        H = self.model_cfg.numHeads
        D = self.model_cfg.headDim

        for l, focalSize in enumerate(focalSizes):

            mac = 0

            # MACS from attention calculation

            # projection in
            mac += nW * (1 + windowSize) * C * H * D * 3

            # attention, softmax is omitted

            mac += 2 * nW * H * D * (1 + windowSize) * (1 + focalSize)

            # projection out

            mac += nW * (1 + windowSize) * C * H * D

            # MACS from MLP layer (2 layers with expand ratio = 1)

            mac += 2 * (T + nW) * C * C

            macs.append(mac)

        return macs, np.sum(macs) * 2  # FLOPS = 2 * MAC

    def forward(self, roiSignals, analysis=False):
        """
        Input :

            roiSignals : (batchSize, N, dynamicLength)

            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise

        Output:

            logits : (batchSize, #ofClasses)


        """
        if self.model_cfg.tokens == "timeseries":
            roiSignals = roiSignals[0].permute((0, 2, 1))
        else:
            roiSignals = roiSignals[0]
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
            logits = self.classifierHead(cls.mean(dim=1))  # (batchSize, #ofClasses)
        elif self.model_cfg.pooling == "gmp":  # pragma: no cover
            logits = self.classifierHead(roiSignals.mean(dim=1))

        torch.cuda.empty_cache()

        return logits, cls
    