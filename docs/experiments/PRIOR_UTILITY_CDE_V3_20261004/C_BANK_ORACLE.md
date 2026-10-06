# C bank oracle

仅限训练后固定5000步bank，先跨三个噪声求平均，再逐图取最优。oracle不可部署，也不支持混合输出上界。

```json
{
  "finetune_seed": 20261004,
  "bank_step": 5000,
  "oracle_psnr": 25.167520087980353,
  "dev_oracle_fixed": {
    "null": 24.80289036912748,
    "physical": 24.62895137265903,
    "histogram": 24.594158154885342,
    "highfreq": 24.601134273365055,
    "all": 24.605859980162716
  },
  "oracle_minus_dev_fixed": 0.3646297188528749,
  "oracle_minus_base": 0.5296133381184447,
  "rgb_psnr": 24.636862655474584,
  "mode_psnr_spread": 0.9296682239846992,
  "pass": true,
  "status": "space_available_not_selector_evidence",
  "oracle_order": "mean over101/102/103 then max over5modes",
  "metrics_complete": true
}
```
