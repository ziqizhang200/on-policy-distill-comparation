"""离线验收本机 CUDA / Qwen3 / LoRA 基础环境，不产生正式实验结果。"""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import bitsandbytes as bnb
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    device = torch.device('cuda:0')
    tensor = torch.randn(32, 32, device=device, requires_grad=True)
    cuda_product = tensor @ tensor.T
    cpu_tensor = tensor.detach().cpu()
    reference = cpu_tensor @ cpu_tensor.T
    assert torch.allclose(cuda_product.detach().cpu(), reference, rtol=1e-4, atol=1e-5)
    loss = cuda_product.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(tensor.grad).all()
    assert torch.cuda.is_bf16_supported(), '当前设备不支持 BF16'
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = tensor.detach() @ tensor.detach().T
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    assert torch.allclose(result.float().cpu(), reference, rtol=3e-2, atol=0.15)
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                        attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    assert trainable and len(trainable) < len(list(model.parameters()))
    before = [param.detach().clone() for param in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    tokens = torch.randint(0, 256, (2, 16), device=device)
    output = model(input_ids=tokens, labels=tokens)
    assert output.logits.shape == (2, 16, 256)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in trainable)
    assert any(param.grad.abs().sum() > 0 for param in trainable)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, trainable))
    quantized_layer = bnb.nn.Linear4bit(64, 32, bias=False,
                                       compute_dtype=torch.bfloat16, quant_type='nf4',
                                       compress_statistics=False).to(device)
    quantized_input = torch.randn(2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    quantized_output = quantized_layer(quantized_input)
    assert quantized_output.shape == (2, 32) and torch.isfinite(quantized_output).all()
    quantized_output.float().square().mean().backward()
    assert quantized_input.grad is not None and torch.isfinite(quantized_input.grad).all()
    assert quantized_input.grad.abs().sum() > 0
    torch.cuda.synchronize()
    packages = ['bitsandbytes', 'torch', 'transformers', 'peft', 'accelerate', 'datasets', 'numpy',
                'scipy', 'pandas', 'matplotlib', 'seaborn', 'PyYAML', 'tqdm',
                'scikit-learn', 'pytest', 'ipykernel', 'jupyterlab', 'tensorboard',
                'sentencepiece', 'protobuf', 'huggingface-hub', 'safetensors', 'tokenizers']
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                           capture_output=True, text=True, check=True)
    report = {'status': 'PASS', 'scope': '环境离线冒烟验收，非正式模型实验',
              'python': platform.python_version(), 'python_executable': sys.executable,
              'platform': platform.platform(), 'cudnn': torch.backends.cudnn.version(),
              'gpu': torch.cuda.get_device_name(0),
              'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'torch_cuda_runtime': torch.version.cuda,
              'nf4_linear_forward_input_backward': True,
              'bf16': True, 'cuda_matmul_backward': True,
              'random_qwen3_lora_optimizer_step': True,
              'qwen3_logits_shape': list(output.logits.shape),
              'qwen3_loss': float(output.loss.detach()),
              'pip_check': check.stdout.strip(),
              'packages': {name: importlib.metadata.version(name) for name in packages}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
