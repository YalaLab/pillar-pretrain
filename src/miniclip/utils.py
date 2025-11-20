from abc import ABC, abstractmethod
import numpy as np
from itertools import repeat
import collections.abc
from lifelines.utils.btree import _BTree
import torch
from torch import Tensor

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from torchmetrics.functional import (
    auroc,
    precision_recall_curve,
    auc,
    average_precision,
)
import warnings



def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement (attention/conv blocks can be wired similarly if needed).
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype


class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob



# class AbstractMetric(ABC):
#     def __init__(self, args, **kwargs):
#         self.args = args

#     @abstractmethod
#     def __call__(self, **kwargs) -> dict[str, Tensor]:
#         pass



class SurvivalMetric:
    def __init__(self, dataset_info, max_followup=6, **kwargs):
        super().__init__()

        self.dataset_info = dataset_info
        self.censoring_distribution = dataset_info["censoring_distribution"]
        # logger.info(f"Censoring distribution: {self.censoring_distribution}")

        self.max_followup = max_followup
        # logger.info(f"Max followup: {self.max_followup}")

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds", "censors"]

    def __call__(self, probs=None, preds=None, golds=None, censors=None, **extras):
        """
        Computes standard classification metrics

        Args:
            probs, preds, golds, censors: items from computing loss and model outputs

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            In multiclass setting (>2), accuracy, and micro-f1, micro-recall, micro-precision are equivalent
            Macro: calculates metric per class then averages
        """

        stats_dict = {}

        censor_times, probs, golds = (
            censors,
            probs,
            golds,
        )
        for followup in range(self.max_followup):
            min_followup_if_neg = followup + 1
            roc_auc, ap_score, pr_auc = compute_auc_at_followup(
                probs, censor_times, golds, followup
            )
            stats_dict["{}_year_auc".format(min_followup_if_neg)] = roc_auc
            stats_dict["{}_year_apscore".format(min_followup_if_neg)] = ap_score
            stats_dict["{}_year_prauc".format(min_followup_if_neg)] = pr_auc

        if golds.sum() > 0:
            cindex = concordance_index(
                censor_times.cpu().numpy(),
                probs.cpu().numpy(),
                golds.cpu().numpy(),
                self.censoring_distribution,
            )
            stats_dict["c_index"] = torch.tensor(cindex, device=probs.device)
        else:
            stats_dict["c_index"] = -1.0
        return stats_dict


def compute_auc_at_followup(probs, censor_times, golds, followup, fup_lower_bound=-1):
    if len(probs.shape) == 3:
        probs = probs.reshape(probs.shape[0] * probs.shape[1], probs.shape[2])

    def include_exam_and_determine_label(prob_arr, censor_time, gold):
        valid_pos = gold and censor_time <= followup and censor_time > fup_lower_bound
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(prob_arr, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(label)

    probs_for_eval = torch.tensor(probs_for_eval, device=probs.device)
    golds_for_eval = torch.tensor(golds_for_eval, device=probs.device)

    try:
        roc_auc = auroc(probs_for_eval, golds_for_eval, pos_label=1, num_classes=2)

        ap_score = average_precision(probs_for_eval, golds_for_eval, pos_label=1)
        precision, recall, _ = precision_recall_curve(
            probs_for_eval, golds_for_eval, pos_label=1
        )
        pr_auc = auc(recall, precision)
    except Exception as e:
        warnings.warn("Failed to calculate AUC because {}".format(e))
        roc_auc = -1.0
        ap_score = -1.0
        pr_auc = -1.0
    return roc_auc, ap_score, pr_auc


def concordance_index(
    event_times, predicted_scores, event_observed=None, censoring_dist=None
):
    r"""
    ## Adapted from: https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/utils/concordance.py
    ## Modified to weight by ipcw (inverse probality of censor weight) to fit Uno's C-index
    ## Modified to use a time-dependent score

    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The c-index is the average of how often a model says X is greater than Y when, in the observed
    data, X is indeed greater than Y. The c-index also handles how to handle censored values
    (obviously, if Y is censored, it's hard to know if X is truly greater than Y).


    The concordance index is a value between 0 and 1 where:

    - 0.5 is the expected result from random predictions,
    - 1.0 is perfect concordance and,
    - 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Parameters:
        event_times: (iterable)
            a length-n iterable of observed survival times.
        predicted_scores: (iterable)
            a length-n iterable of predicted scores - these could be survival times, or hazards, etc. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        event_observed: (iterable, optional)
            a length-n iterable censorship flags, 1 if observed, 0 if not. Default None assumes all observed.

    Returns:
        c-index: (float) a value between 0 and 1.

    References:
        Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
        developing models, evaluating assumptions and adequacy, and measuring and
        reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Examples:
        >>> from lifelines.utils import concordance_index
        >>> cph = CoxPHFitter().fit(df, 'T', 'E')
        >>> concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
    """
    event_times = np.array(event_times).ravel()
    predicted_scores = 1 - np.asarray(predicted_scores, dtype=float)
    if len(predicted_scores.shape) == 3:
        predicted_scores = predicted_scores.reshape(
            [
                predicted_scores.shape[0] * predicted_scores.shape[1],
                predicted_scores.shape[2],
            ]
        )

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError(
                "Observed events must be 1-dimensional of same length as event times"
            )

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(
        event_times, predicted_scores, event_observed, censoring_dist
    )

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(
    event_times, predicted_event_times, event_observed, censoring_dist
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    observed_times = set(event_times)

    died_mask = event_observed.astype(bool)
    # event_times may not be sorted, so we explicitly order them here.
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = {}
    for time in observed_times:
        times_to_compare[time] = _BTree(np.unique(died_pred[:, int(time)]))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (
            not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                censored_truth,
                censored_pred,
                censored_ix,
                times_to_compare,
                censoring_dist,
            )
            censored_ix = next_ix
        elif has_more_died and (
            not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                died_truth, died_pred, died_ix, times_to_compare, censoring_dist
            )
            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare, censoring_dist):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns:
      (pairs, correct, tied, next_ix)
        new_pairs: The number of new comparisons performed
        new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    truth_time = truth[first_ix]
    weight = 1.0 / (censoring_dist[str(int(truth_time))] ** 2)
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare[truth_time]) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare[truth_time].rank(pred[i][int(truth_time)])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix, weight)
