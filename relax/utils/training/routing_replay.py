import os

import torch

from relax.utils import device as device_utils
from relax.utils.env import Envs


ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(device_utils.make_current_torch_device())

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(device_utils.make_current_torch_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def _align_top_indices(top_indices, scores_len):
    """Slice or pad top_indices along dim-0 to match scores_len.

    VLM bridge models with CP may use different sequence alignment than
    fill_routing_replay, causing a padding mismatch.  The extra (or missing)
    entries correspond to padding tokens that are zeroed out by the loss mask,
    so truncating or zero-padding is safe.
    """
    if top_indices.shape[0] == scores_len:
        return top_indices
    if top_indices.shape[0] > scores_len:
        return top_indices[:scores_len]
    pad_rows = scores_len - top_indices.shape[0]
    return torch.nn.functional.pad(top_indices, (0, 0, 0, pad_rows), value=0)


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # ROUTING_REPLAY is None for routers that opt out of replay (e.g. MTP),
        # in which case we fall through to the original implementation.
        if Envs.ENABLE_ROUTING_REPLAY and ROUTING_REPLAY is not None:
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                top_indices = _align_top_indices(top_indices, scores.shape[0])
                assert top_indices.shape[1] == topk, (
                    f"[{torch.distributed.get_rank()}] top_indices topk {top_indices.shape[1]} does not match expected topk {topk}"
                )
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                top_indices = _align_top_indices(top_indices, scores.shape[0])
                assert top_indices.shape[1] == topk, (
                    f"top_indices topk {top_indices.shape[1]} does not match expected topk {topk}"
                )
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if not Envs.ENABLE_ROUTING_REPLAY:
        return

    # MTP routers exist in training but rollout (sglang) doesn't run MTP, so
    # there's nothing to record/replay against.  Install a pre-hook that clears
    # the global ROUTING_REPLAY (compute_topk falls through) and skip
    # registration in `all_routing_replays` so fill_routing_replay's per-layer
    # accounting stays consistent.
    if getattr(module, "is_mtp_layer", False):

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(None)

        module.register_forward_pre_hook(pre_forward_hook)
        return

    module.routing_replay = RoutingReplay()

    def pre_forward_hook(*args, **kwargs):
        set_routing_replay(module.routing_replay)

    module.register_forward_pre_hook(pre_forward_hook)
