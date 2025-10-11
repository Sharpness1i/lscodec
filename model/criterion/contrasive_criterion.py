import torch
from torch import nn
import torch.nn.functional as F


def buffered_arange(max, device="cpu"):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor().to(device)
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


class ContrasiveCriterion(nn.Module):
    """
    modified from fairseq Wav2vecCriterion
    """
    def __init__(
        self, 
        encoder_embed_dim = 512,
        final_dim = 256,
        num_negatives = 100,  # 负例步数
        cross_sample_negatives = 0,  # 跨样本（batch内）负例步数
        logit_temp = 0.1,
    ):
        super().__init__()
        self.logit_temp = logit_temp
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives

        final_dim = final_dim if final_dim > 0 else encoder_embed_dim
        self.project_y = nn.Linear(encoder_embed_dim, final_dim)
        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
    
    def forward(self, cnn_feat, mask_indices, quantized, reduce=True):
        cnn_feat = cnn_feat.transpose(1,2)  ##[B,T,D]
        y = cnn_feat[mask_indices].view(cnn_feat.size(0), -1, cnn_feat.size(-1))  ##target:[B,mask_T,D]
        y = self.project_y(y)
        negs, _ = self.sample_negatives(y, y.size(1), padding_count=None)  ##negative sample:[N,B,mask_T,D]

        quantized = quantized.transpose(1,2)  ##[B,T,D]
        x = quantized[mask_indices].view(quantized.size(0), -1, quantized.size(-1))  ##x:reconstructed token:[B,mask_T,D]
        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs) #[N+1,B,mask_T]

        logits = self.get_logits(x).float() ##[mask_T,N+1]
        target = self.get_targets(x) #[mask_T]

        # breakpoint()
        loss = F.cross_entropy(logits, target, reduction="sum")

        return loss


    def compute_preds(self, x, y, negatives):
        """
            x: masked quantized codes
            y: unmask cnn feat
            negatives: selected from y
        """

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)  #  [1,B,mask_T,D]
        targets = torch.cat([y, negatives], dim=0)  #  [1+N,B,mask_T,D]

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)  #  [1+N,B,mask_T,D]
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def get_logits(self, x):
        # logits = net_output["x"]
        logits = x.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, x):
        # x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def sample_negatives(self, y, num, padding_count=None):

        if self.num_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.num_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.num_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.num_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.num_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.num_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
        
        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.num_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


if __name__ == '__main__':

    from typing import Optional, Tuple
    import numpy as np
    def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
        require_same_masks: bool = True,
        mask_dropout: float = 0.0,
        add_masks: bool = False,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
        idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
        num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
    ) -> np.ndarray:
        """
        Computes random mask spans for a given shape

        Args:
            shape: the the shape for which to compute masks.
                should be of size 2 where first element is batch size and 2nd is timesteps
            padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
            mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
                number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
                however due to overlaps, the actual number will be smaller (unless no_overlap is True)
            mask_type: how to compute mask lengths
                static = fixed size
                uniform = sample from uniform distribution [mask_other, mask_length*2]
                normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
                poisson = sample from possion distribution with lambda = mask length
            min_masks: minimum number of masked spans
            no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
            min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
            require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
            mask_dropout: randomly dropout this percentage of masks in each example
        """

        bsz, all_sz = shape
        mask = np.full((bsz, all_sz), False)

        if num_mask_ver == 1:
            all_num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * all_sz / float(mask_length)
                + np.random.rand()
            )
            all_num_mask = max(min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):
            if seed is not None and epoch is not None and indices is not None:
                seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
            else:
                seed_i = None

            rng = np.random.default_rng(seed_i)

            if padding_mask is not None:
                sz = all_sz - padding_mask[i].long().sum().item()
                assert sz >= 0, sz
            else:
                sz = all_sz

            if num_mask_ver == 1:
                if padding_mask is not None:
                    num_mask = int(
                        # add a random number for probabilistic rounding
                        mask_prob * sz / float(mask_length)
                        + np.random.rand()
                    )
                    num_mask = max(min_masks, num_mask)
                else:
                    num_mask = all_num_mask
            elif num_mask_ver == 2:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + rng.random()
                )
                num_mask = max(min_masks, num_mask)
            else:
                raise ValueError()

            if mask_type == "static":
                lengths = np.full(num_mask, mask_length)
            elif mask_type == "uniform":
                lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
            elif mask_type == "normal":
                lengths = rng.normal(mask_length, mask_other, size=num_mask)
                lengths = [max(1, int(round(x))) for x in lengths]
            elif mask_type == "poisson":
                lengths = rng.poisson(mask_length, size=num_mask)
                lengths = [int(round(x)) for x in lengths]
            else:
                raise Exception("unknown mask selection " + mask_type)

            if sum(lengths) == 0:
                if mask_type == "static":
                    raise ValueError(f"this should never happens")
                else:
                    lengths = [min(mask_length, sz - 1)]

            if no_overlap:
                mask_idc = []

                def arrange(s, e, length, keep_length):
                    span_start = rng.randint(s, e - length)
                    mask_idc.extend(span_start + i for i in range(length))

                    new_parts = []
                    if span_start - s - min_space >= keep_length:
                        new_parts.append((s, span_start - min_space + 1))
                    if e - span_start - length - min_space > keep_length:
                        new_parts.append((span_start + length + min_space, e))
                    return new_parts

                parts = [(0, sz)]
                min_length = min(lengths)
                for length in sorted(lengths, reverse=True):
                    lens = np.fromiter(
                        (e - s if e - s >= length + min_space else 0 for s, e in parts),
                        np.int,
                    )
                    l_sum = np.sum(lens)
                    if l_sum == 0:
                        break
                    probs = lens / np.sum(lens)
                    c = rng.choice(len(parts), p=probs)
                    s, e = parts.pop(c)
                    parts.extend(arrange(s, e, length, min_length))
                mask_idc = np.asarray(mask_idc)
            else:
                if idc_select_ver == 1:
                    min_len = min(lengths)
                    if sz - min_len <= num_mask:
                        min_len = sz - num_mask - 1
                    mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
                elif idc_select_ver == 2:
                    mask_idc = rng.choice(sz, num_mask, replace=False)
                else:
                    raise ValueError()

                mask_idc = np.asarray(
                    [
                        mask_idc[j] + offset
                        for j in range(len(mask_idc))
                        for offset in range(lengths[j])
                    ]
                )

            mask_idc = np.unique(mask_idc[mask_idc < sz])
            if len(mask_idc) >= sz:
                raise ValueError(
                    (
                        f"the entire sequence is masked. "
                        f"sz={sz}; mask_idc[mask_idc]; "
                        f"index={indices[i] if indices is not None else None}"
                    )
                )
            mask_idcs.append(mask_idc)

        target_len = None
        if require_same_masks:
            if add_masks:
                target_len = max([len(m) for m in mask_idcs])
            else:
                target_len = min([len(m) for m in mask_idcs])

        for i, mask_idc in enumerate(mask_idcs):
            if target_len is not None and len(mask_idc) > target_len:
                mask_idc = rng.choice(mask_idc, target_len, replace=False)

            mask[i, mask_idc] = True

            if target_len is not None and len(mask_idc) < target_len:
                unmasked = np.flatnonzero(~mask[i])
                to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
                mask[i, to_mask] = True

            if mask_dropout > 0:
                masked = np.flatnonzero(mask[i])
                num_holes = np.rint(len(masked) * mask_dropout).astype(int)
                to_drop = rng.choice(masked, num_holes, replace=False)
                mask[i, to_drop] = False

        return mask


    def apply_mask(
        x, 
        # temperal mask
        mask_prob=0.1,  #  "probability of replacing a token with mask"
        mask_length=3,  # "mask length"
        mask_selection="static",  # "how to choose mask length"
        mask_other=0, # "secondary mask argument (used for more complex distributions), see help in compute_mask_indices"
        no_mask_overlap=False, # "whether to allow masks to overlap"
        mask_min_space=1, # "min space between spans (if no overlap is enabled)"
        # channel mask
        mask_channel_prob=0.0, # "probability of replacing a feature with 0"
        mask_channel_length=10, # "length of the mask for features (channels)"
        mask_channel_selection="static", # "how to choose mask length for channel masking"
        mask_channel_other=0, # "secondary mask argument (used for more complex distributions), see help in compute_mask_indices"
        no_mask_channel_overlap=False, # "whether to allow channel masks to overlap"
        mask_channel_min_space=1, # "min space between spans (if no overlap is enabled)"
        padding_mask=None,
        target_list=None
    ):
        mask_emb = nn.Parameter(
            torch.FloatTensor(x.size(-1)).uniform_()
        )  # 可学习的参数，应初始化在model里

        B, T, C = x.shape
        if mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                mask_selection,
                mask_other,
                min_masks=2,
                no_overlap=no_mask_overlap,
                min_space=mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                mask_channel_prob,
                mask_channel_length,
                mask_channel_selection,
                mask_channel_other,
                no_overlap=no_mask_channel_overlap,
                min_space=mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0
        return x, mask_indices
    
    
    
    cnn_feat = torch.randn(4, 313, 512)  # (b,t,d)
    masked_cnn_feat, mask_indices = apply_mask(cnn_feat.clone())  # 内部为replace操作，需要clone
    
    loss = ContrasiveCriterion()(
        cnn_feat.transpose(1, 2),  # 给定没有mask的feat
        mask_indices,  # mask的index
        cnn_feat.transpose(1, 2),  # 应该给定量化后的feat, (B,D,T)
    )

    print(loss)

