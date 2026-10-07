"""Synthetic CUDA-extension checks on the actual vendor device."""
import time
import torch
from .budget import DeviceBudget
from .records import write,sha


def probe(run):
    import causal_conv1d,causal_conv1d_cuda,selective_scan_cuda,mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref
    result={'kind':'EXTENSION_ONLY_NOT_BACKBONE','device':torch.cuda.get_device_name(0),
            'mamba_version':mamba_ssm.__version__,'scan_binary_sha256':sha(selective_scan_cuda.__file__),
            'causal_binary_sha256':sha(causal_conv1d_cuda.__file__)}
    with DeviceBudget(run) as b:
        b.start('selective_scan_and_causal_real_PPU_validation',120)
        torch.manual_seed(12)
        u=torch.randn(2,16,64,device='cuda');delta=torch.randn_like(u)*.1
        A=-torch.rand(16,8,device='cuda')
        B=torch.randn(2,4,8,64,device='cuda');C=torch.randn_like(B);D=torch.ones(16,device='cuda')
        torch.cuda.synchronize();begin=time.monotonic()
        with torch.no_grad():
            actual=selective_scan_fn(u,delta,A,B,C,D,delta_softplus=True)
            ref=selective_scan_ref(u,delta,A,B,C,D,delta_softplus=True)
            scan_error=float((actual-ref).abs().max())
            x=torch.randn(2,16,64,device='cuda');weight=torch.randn(16,3,device='cuda');bias=torch.randn(16,device='cuda')
            conv=causal_conv1d.causal_conv1d_fn(x,weight,bias,activation='silu')
            truth=torch.nn.functional.silu(torch.nn.functional.conv1d(x,weight[:,None],bias,padding=2,groups=16)[...,:64])
            conv_error=float((conv-truth).abs().max())
        torch.cuda.synchronize()
        result.update(scan_max_abs_error=scan_error,causal_max_abs_error=conv_error,seconds=time.monotonic()-begin,
                      passed=bool(torch.allclose(actual,ref,atol=1e-4,rtol=1e-4) and torch.allclose(conv,truth,atol=1e-4,rtol=1e-4)))
    write(run/'tests/runtime_probe.json',result)
    if not result['passed']:raise RuntimeError('real PPU extension validation failed')
    return result
