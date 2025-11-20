from typing import Dict, Optional, Union


import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            **kwargs
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
            **kwargs,
    ):
        total_loss = 0
        device = image_features.device
        if isinstance(text_features, dict):
            for text_category, text_feat in text_features.items():
                logits_per_image, logits_per_text = self.get_logits(
                    image_features,
                    text_feat,
                    logit_scale,
                    logit_bias=logit_bias,
                )
                labels = self.get_ground_truth(device, logits_per_image.shape[0])
                total_loss += (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2
        else:
            logits_per_image, logits_per_text = self.get_logits(
                image_features,
                text_features,
                logit_scale,
                logit_bias=logit_bias,
            )

            labels = self.get_ground_truth(device, logits_per_image.shape[0])

            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """
    Implements the Sigmoid Loss for Language-Image Pre-Training (SigLIP).

    This loss correctly handles both single-GPU and distributed training. It is also
    extensible: if `text_features` is a dictionary of tensors, it computes the
    loss for each and averages the results.

    Reference: https://arxiv.org/abs/2303.15343

    Args:
        cache_labels (bool): If True, caches ground-truth labels for efficiency.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    def __init__(self, cache_labels: bool = True, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.labels = {}
        self.prev_num_logits = 0

    def get_logits(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        logit_scale: torch.Tensor, 
        logit_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Computes the image-text similarity logits."""
        logits = logit_scale * (image_features @ text_features.T)
        if logit_bias is not None:
            logits += logit_bias
        ## print the min and max of logits
        print(f"Min logits: {logits.min()}, Max logits: {logits.max()}")
        return logits

    def _calculate_loss(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Helper function to compute the loss for a single pair of feature tensors."""
        device = logits.device
        num_logits = logits.shape[0]

        # Get or create cached ground-truth labels
        if self.cache_labels and self.prev_num_logits == num_logits and device in self.labels:
            labels = self.labels[device]
        else:
            labels = 2 * torch.eye(num_logits, device=device, dtype=logits.dtype) - 1
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        
        # Pairwise sigmoid loss calculation
        loss = -F.logsigmoid(labels * logits).sum() / num_logits
        return loss

    def forward(
        self,
        logits: torch.Tensor,
        output_dict: bool = False,
    ):
        """
        Calculates the SigLIP loss.

        Dispatches to the appropriate logic based on the type of `text_features`.
        """
        if isinstance(logits, dict):
            all_losses = [
                self._calculate_loss(logits[text_category])
                for text_category in logits.keys()
            ]
            final_loss = torch.stack(all_losses).mean()
        else:
            final_loss = self._calculate_loss(logits)
        return {"contrastive_loss": final_loss} if output_dict else final_loss


class TextDinoLoss(nn.Module):
    """
    Implements the TextDino loss - DINO-style distillation from text teacher to vision student.
    
    Key features:
    - Cross-entropy loss between teacher and student outputs
    - Teacher centering to prevent mode collapse
    - Different temperature scaling for teacher (sharp) and student (soft)
    
    Args:
        cache_labels (bool): If True, caches ground-truth labels for efficiency.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    def __init__(self, cache_labels: bool = True, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        
    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor, 
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
        output_dict: bool = False,
    ):
        """
        Calculate TextDino loss.
        
        Args:
            teacher_logits: Teacher output after temperature scaling
            student_logits: Student output after temperature scaling
            center: Center for teacher outputs (prevents collapse)
            teacher_temp: Temperature for teacher (should be low)
            student_temp: Temperature for student (should be higher)
            output_dict: If True, return dict with loss components
            
        Returns:
            Loss value or dict with loss components
        """
        # Apply centering to teacher (only teacher gets centered)
        teacher_centered = teacher_logits - center
        
        # Apply softmax with respective temperatures
        # Note: temperatures are already applied in the model, so we just do softmax
        teacher_probs = F.softmax(teacher_centered, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        # Calculate cross-entropy loss
        # Using KL divergence: -sum(P_teacher * log(P_student))
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        
        if output_dict:
            return {
                "distillation_loss": loss,
                "teacher_entropy": -(teacher_probs * teacher_probs.log()).sum(dim=-1).mean(),
                "student_entropy": -(student_log_probs.exp() * student_log_probs).sum(dim=-1).mean(),
            }
        
        return loss