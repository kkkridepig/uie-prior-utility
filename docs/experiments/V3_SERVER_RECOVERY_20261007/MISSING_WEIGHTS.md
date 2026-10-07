# 当前缺失的历史自训权重

以下 11 项来自断电前 V3 `closeout/weights.csv`。本轮逐一检查均不存在。旧表中的加载验收属于历史，本轮未重新验收这些权重。V2/S2 的其他自训 checkpoint 也未出现在新服务器中；本表只列 V3 收束包明确登记的权重，不冒充整个旧服务器的完整权重清单。

所有路径均相对于 `/mnt/workspace/uie-prior-utility`。V3 的分支是 delta，找回分支仍需同时找回对应父权重；只剩 delta 不能正确推理。

| 名称 | 原路径 | 原 bytes | 原 SHA256 |
| --- | --- | ---: | --- |
| PARENT，85000 步 | `runs/explore_ag_single_seed_v2_20261003/parent_0.pt` | 171492582 | `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea` |
| BASE_CONT_V3 | `runs/prior_utility_cde_v3_20261004/pilot/20261004/BASE_CONT_V3/step_05000.pt` | 72218941 | `91a465dac86d81deab918a6595f1d8a81140616766086edca76d81455d6e1fb3` |
| C_BANK | `runs/prior_utility_cde_v3_20261004/pilot/20261004/C_BANK/step_05000.pt` | 1876077 | `643bb9435eea323ddca2282efc5c2338d8d9ea36d63a754c9eefd7f246b43556` |
| C_RGB_CONTROL | `runs/prior_utility_cde_v3_20261004/pilot/20261004/C_RGB_CONTROL/step_05000.pt` | 1876077 | `96a520a124fefd13885c855d4cdbd99291042332976caf8321692eef2e955a02` |
| C_ALL_ONLY | `runs/prior_utility_cde_v3_20261004/pilot/20261004/C_ALL_ONLY/step_05000.pt` | 1876013 | `c783abfc1d3e4e3adc90837520cfad41e942252c04c7e42295916cf59d9ac851` |
| D_PHASE_STD | `runs/prior_utility_cde_v3_20261004/pilot/20261004/D_PHASE_STD/step_05000.pt` | 132923 | `55f300526d65fca0889463af072c36405825875d769ffbddd3b26194c1ed2482` |
| D_SOBEL_STD | `runs/prior_utility_cde_v3_20261004/pilot/20261004/D_SOBEL_STD/step_05000.pt` | 132923 | `3c33c5d24e5acbcd05879c08affe0469acc7a214b6531d1f8a3e20ebf7b8e5d7` |
| D_SOFTPHASE_STD | `runs/prior_utility_cde_v3_20261004/pilot/20261004/D_SOFTPHASE_STD/step_05000.pt` | 132923 | `f0e68dec17b17305e1088de21f0afcf3cf979796609aaf915f49cfdeb8d167b4` |
| SHUFFLED selector | `runs/prior_utility_cde_v3_20261004/pilot/20261004/selectors/SHUFFLED/last.pt` | 469221 | `97c5f05e414d930b59df1a35acc43422ddcb0e042621970e638d0b4cabc346ca` |
| UTILITY selector | `runs/prior_utility_cde_v3_20261004/pilot/20261004/selectors/UTILITY/last.pt` | 469221 | `909406b13992bba5d9e3a3e33fe2dc33d476f41f425c1c0e5dce533fefc040bb` |
| WINNER_CE selector | `runs/prior_utility_cde_v3_20261004/pilot/20261004/selectors/WINNER_CE/last.pt` | 469221 | `652be1f8cfa1f2803b08f5c2f10089aa43fc1c712fcd67df8eb78c26dcc25d61` |

公开 DA/VGG 权重已经重新下载，详见 [RECOVERY_REPORT.md](RECOVERY_REPORT.md)。它们不包含上述自训模型的参数。
