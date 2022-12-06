# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
from .fixmatch import FixMatch


class MaxMatch(FixMatch):
    def __init__(self, base_encoder, num_classes=1000, eman=False, momentum=0.999, norm=None):
        super(MaxMatch, self).__init__(base_encoder=base_encoder,
                                       num_classes=num_classes, eman=eman, momentum=momentum, norm=norm)

    def forward(self, im_x, im_u_w=None, im_u_s=None):
        if im_u_w is None and im_u_s is None:
            logits = self.main(im_x)
            return logits

        # K = im_u_s.shape[0]  # bangyan
        # im_u_s_K = torch.cat([im_u_s[i] for i in range(K)])  # bangyan
        batch_size_x = im_x.shape[0]
        batch_size_u = im_u_w.shape[0]
        if not self.eman:
            inputs = torch.cat((im_x, im_u_w, im_u_s))  # bangyan
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_w, logits_u_s = \
                logits[batch_size_x:batch_size_x + batch_size_u], \
                logits[batch_size_x + batch_size_u:]  # bangyan
        else:
            # use ema model for pesudo labels
            inputs = torch.cat((im_x, im_u_s))  # bangyan
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_s = logits[batch_size_x:]
            with torch.no_grad():  # no gradient to ema model
                logits_u_w = self.ema(im_u_w)

        return logits_x, logits_u_w, logits_u_s


def get_maxmatch_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        MaxMatch model
    """
    if isinstance(model, str):
        model = {
            "MaxMatch": MaxMatch,
        }[model]
    return model
