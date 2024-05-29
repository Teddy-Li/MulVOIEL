from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.distributed as dist
from pkg_resources import packaging
import torch.cuda.nccl as nccl
from config.mixed_precision import bfSixteen, fpSixteen, bfSixteen_mixed
import functools
import os
from transformers import TrainerCallback
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, _or_policy, lambda_auto_wrap_policy
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder


def lora_wrap(model, lora_r, lora_alpha, lora_dropout):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout)  #, target_modules=["q", "k", "v", "o"])  # TODO: check target_modules
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def set_profiler(profile_path='./logs/tensorboard'):
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat
    )
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()
    profiler_cb = ProfilerCallback(profiler)
    return profiler, profiler_cb


# borrowed from https://github.com/facebookresearch/llama-recipes/blob/main/policies/wrapping.py
def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy


# borrowed from https://github.com/facebookresearch/llama-recipes/blob/main/utils/train_utils.py
def get_policies(mixed_precision: bool, use_fp16: bool, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready and not use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


# borrowed from https://github.com/facebookresearch/llama-recipes/blob/main/utils/fsdp_utils.py
def fsdp_auto_wrap_policy():
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            LlamaDecoderLayer,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )
    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


# borrowed from https://github.com/facebookresearch/llama-recipes/blob/main/utils/train_utils.py
def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


# borrowed from https://github.com/facebookresearch/llama-recipes/blob/main/utils/train_utils.py
def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def compute_metrics(metric, predictions, labels):
    if metric == 'accuracy':
        return (predictions == labels).sum().item() / len(labels)
    elif metric == 'f1':
        return f1_score(labels, predictions, average='macro')
    elif metric == 'precision':
        return precision_score(labels, predictions, average='macro')
    elif metric == 'recall':
        return recall_score(labels, predictions, average='macro')
    else:
        raise ValueError(f"Unknown metric {metric}")
