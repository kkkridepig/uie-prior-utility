# Baseline verification

SS-UIE official public simplified implementation; commit 88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df
Weight SHA256: 977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99

| Policy | model_val mean PSNR dB |
| --- | --- |
| clip01 | 27.523331725318602 |
| official_minmax_float | 25.67596423880748 |

Policy selection only on 671 model_val images; frozen clip01. This is not a reproduction of the full AAAI paper model or author Test-400 scores.
Source provenance: backbone_provenance.json; strict uploaded-weight PPU receipt; author issue-comment evidence.
Metric identity: baseline_policy.json (PSNR/SSIM and actual pretrained VGG LPIPS). Fixed 256x256 RGB float32 preprocessing.
