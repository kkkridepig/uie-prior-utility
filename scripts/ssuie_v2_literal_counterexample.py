"""CPU float64 receipt for the protocol's exact two-region counterexample."""
import time
import torch
from uie_next.records import ROOT, sha, write
from uie_next.v2.context import RUN_ID
from uie_next.v2.diagnostics import exact_strategies


def main():
    begin=time.monotonic()
    base=torch.full((1,3,256,256),.4,dtype=torch.float64)
    target=torch.full_like(base,.5)
    candidate=torch.full_like(base,.3)
    candidate[...,:128]=.5
    variants,error=exact_strategies(base,candidate,target)
    def psnr(output):
        return float(-10*torch.log10((output-target).square().mean()))
    observed={k:psnr(variants[k][0]) for k in ['B0','endpoint','oracle_pixel']}
    expected={'B0':20.,'endpoint':16.989700043360187,'oracle_pixel':23.010299956639813}
    assert max(abs(observed[k]-v) for k,v in expected.items())<1e-10
    quadratic_error=0.
    for alpha in [0,.25,.5,.75,1]:
        mse=float((variants['fixed_'+str(alpha)][0]-target).square().mean())
        quadratic_error=max(quadratic_error,abs(mse-.01*(1+alpha**2)))
    assert quadratic_error<1e-10
    write(ROOT/'runs'/RUN_ID/'tests/literal_protocol_counterexample.json',
          {'passed':True,'synthetic_fixture_only':True,'float64':True,'GPU_device_seconds':0,
           'script_sha256':sha(__file__),'observed_psnr':observed,'expected_psnr':expected,
           'global_strength_mse_identity_max_error':quadratic_error,'oracle_order_max_error':error,
           'CPU_wall_seconds':time.monotonic()-begin})
    print(observed)


if __name__=='__main__':main()
