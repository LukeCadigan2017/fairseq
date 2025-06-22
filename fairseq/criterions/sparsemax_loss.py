# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from entmax import SparsemaxLoss, sparsemax_bisect_loss

@dataclass
class SparsemaxCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )

@register_criterion("sparsemax_loss", dataclass=SparsemaxCriterionConfig)
class SparsemaxCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        # self.sparsemax_loss=sparsemax_bisect_loss
        self.sparsemax_loss=SparsemaxLoss().loss

    

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        # logging_output = {
        #     "loss": loss.data,
        #     "ntokens": sample["ntokens"],
        #     "nsentences": sample["target"].size(0),
        #     "sample_size": sample_size,
        # }

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        #  27371.708984375


        logits=net_output[0]
        X=logits.view(-1, logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        raw_loss=self.sparsemax_loss(X, target)
        

        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction="sum" if reduce else "none",
        # )


        # target size: torch.Size([3072])
        # net_output[0] size: torch.Size([192, 16, 6632])
        # lprobs size: torch.Size([3072, 6632])
        # print(f"target size: {target.size()}")
        # print(f"net_output[0] size: {net_output[0].size()}")
        # print(f"lprobs size: {lprobs.size()}")
        #loss should ideally be around loss is 27371.708984375
        loss=sum(raw_loss)
        # print(f"loss is {loss}")
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
