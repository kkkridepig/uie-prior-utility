# D1–D7证据索引

所有逐图表、梯度、折清单、完整ridge拟合与布尔闸门位于runs/ssuie_utility_v3_predictability_20261007/diagnostics/。D1/D2/D3含PSNR/SSIM/LPIPS。D6/D7是二次式PSNR/误差诊断，未将oracle列作可部署排名。配对区间5000次、seed20261017，内容组重采样后图像等权。

|集合|策略|PSNR|相对J0|相对FIXED_FIT|损害比例|
|---|---|---:|---:|---:|---:|
|oof|FIXED_FIT|23.738357|+0.009125|+0.000000|0.187359|
|oof|RIDGE_GLOBAL|23.680489|-0.048742|-0.057868|0.275395|
|oof|RIDGE_REGION|23.669555|-0.059677|-0.068802|0.322799|
|utility_val|FIXED_FIT|24.372912|+0.017026|+0.000000|0.257353|
|utility_val|RIDGE_GLOBAL|24.245803|-0.110082|-0.127109|0.308824|
|utility_val|RIDGE_REGION|24.233943|-0.121943|-0.138969|0.367647|

完整D1–D6汇总见D1_D6_summary.json，配对区间见metrics/paired_intervals.json。
