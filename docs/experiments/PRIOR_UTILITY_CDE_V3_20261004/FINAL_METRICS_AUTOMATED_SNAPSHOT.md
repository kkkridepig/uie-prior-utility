# 冻结后指标

pilot_no_supported_gain

```json
{
  "status": "pilot_no_supported_gain",
  "development_status": "pilot_no_supported_gain",
  "method": null,
  "per_seed_datasets": {
    "20261004": {
      "LSUI_legacy_exposed_regression": {
        "models": {
          "PARENT": {
            "psnr": 21.51003134499015,
            "ssim": 0.8668702414345327,
            "lpips": 0.21541757435391742,
            "model_seconds": 0.21885015257720739
          },
          "BASE_CONT_V3": {
            "psnr": 22.2785996139781,
            "ssim": 0.8742040737148896,
            "lpips": 0.21127869934134424,
            "model_seconds": 0.21925856269696825
          }
        },
        "comparisons": {
          "PARENT": {
            "psnr": {
              "images": 427,
              "scene_groups": 425,
              "mean": 0.7685682689879506,
              "scene_equal_mean": 0.770047894680192,
              "median": 0.6983775415564158,
              "improved_fraction": 0.7447306791569087,
              "scene_bootstrap_ci95": [
                0.6055157243495792,
                0.9262459727811319
              ],
              "remove_largest5_mean": 0.6995832638936468,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "ssim": {
              "images": 427,
              "scene_groups": 425,
              "mean": 0.0073338322803568985,
              "scene_equal_mean": 0.0073400150262380075,
              "median": 0.006053270126676484,
              "improved_fraction": 0.7049180327868853,
              "scene_bootstrap_ci95": [
                0.0053010409797007866,
                0.009243051087903348
              ],
              "remove_largest5_mean": 0.006378041412923782,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "lpips": {
              "images": 427,
              "scene_groups": 425,
              "mean": -0.004138875012573183,
              "scene_equal_mean": -0.004115430801814677,
              "median": -0.003366991877555847,
              "improved_fraction": 0.41217798594847777,
              "scene_bootstrap_ci95": [
                -0.0057522808327582645,
                -0.002349375974724035
              ],
              "remove_largest5_mean": -0.004837815938179057,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            }
          }
        },
        "dataset_role": "LSUI_legacy_exposed_regression",
        "freeze_sha256": "e11f752ab120580293dc7964230e17e29b58db0fda99581a8e72dd9b994e31ac"
      },
      "UIEB_legacy_exposed_regression": {
        "models": {
          "PARENT": {
            "psnr": 24.380012356461535,
            "ssim": 0.9300233149401704,
            "lpips": 0.09728071076917075,
            "model_seconds": 0.2110751135710025
          },
          "BASE_CONT_V3": {
            "psnr": 24.92194813940931,
            "ssim": 0.9365631216090057,
            "lpips": 0.08648472474855963,
            "model_seconds": 0.21898585843255175
          }
        },
        "comparisons": {
          "PARENT": {
            "psnr": {
              "images": 97,
              "scene_groups": 96,
              "mean": 0.5419357829477712,
              "scene_equal_mean": 0.5264211732875207,
              "median": 0.5341189037946824,
              "improved_fraction": 0.6185567010309279,
              "scene_bootstrap_ci95": [
                0.12927029526315373,
                0.9693985608593947
              ],
              "remove_largest5_mean": 0.264387381821856,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "ssim": {
              "images": 97,
              "scene_groups": 96,
              "mean": 0.006539806668835143,
              "scene_equal_mean": 0.006537052262031856,
              "median": 0.004580695892306956,
              "improved_fraction": 0.6804123711340206,
              "scene_bootstrap_ci95": [
                0.003040707842142816,
                0.010011603801297314
              ],
              "remove_largest5_mean": 0.0041330067168526136,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "lpips": {
              "images": 97,
              "scene_groups": 96,
              "mean": -0.010795986020611117,
              "scene_equal_mean": -0.010776317701028245,
              "median": -0.007552350560824077,
              "improved_fraction": 0.29896907216494845,
              "scene_bootstrap_ci95": [
                -0.015718993097947446,
                -0.006736860532464103
              ],
              "remove_largest5_mean": -0.012850724946896453,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            }
          }
        },
        "dataset_role": "UIEB_legacy_exposed_regression",
        "freeze_sha256": "e11f752ab120580293dc7964230e17e29b58db0fda99581a8e72dd9b994e31ac"
      },
      "confirm_holdout": {
        "models": {
          "PARENT": {
            "psnr": 20.986554293378546,
            "ssim": 0.849667080223481,
            "lpips": 0.2350512510522857,
            "model_seconds": 0.2190549831999385
          },
          "BASE_CONT_V3": {
            "psnr": 21.74829751576583,
            "ssim": 0.8557241133678708,
            "lpips": 0.2304984131273762,
            "model_seconds": 0.21523135083585418
          }
        },
        "comparisons": {
          "PARENT": {
            "psnr": {
              "images": 391,
              "scene_groups": 389,
              "mean": 0.7617432223872866,
              "scene_equal_mean": 0.763299469341636,
              "median": 0.6583892241693311,
              "improved_fraction": 0.731457800511509,
              "scene_bootstrap_ci95": [
                0.6190492651785809,
                0.9136803609534634
              ],
              "remove_largest5_mean": 0.6841748660179519,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "ssim": {
              "images": 391,
              "scene_groups": 389,
              "mean": 0.0060570331443897995,
              "scene_equal_mean": 0.005890192688917529,
              "median": 0.005616597719704131,
              "improved_fraction": 0.7135549872122762,
              "scene_bootstrap_ci95": [
                0.004180517061633949,
                0.007908236044121114
              ],
              "remove_largest5_mean": 0.0051637970540477,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            },
            "lpips": {
              "images": 391,
              "scene_groups": 389,
              "mean": -0.00455283792490951,
              "scene_equal_mean": -0.004600351441718619,
              "median": -0.003918579469124476,
              "improved_fraction": 0.3938618925831202,
              "scene_bootstrap_ci95": [
                -0.006531909591508934,
                -0.002648674505791137
              ],
              "remove_largest5_mean": -0.0053859773431365774,
              "resampling": "scene_group1000_image_weighted; one pretrained parent"
            }
          }
        },
        "dataset_role": "confirm_holdout",
        "freeze_sha256": "e11f752ab120580293dc7964230e17e29b58db0fda99581a8e72dd9b994e31ac"
      }
    }
  },
  "new_holdout_multiseed_psnr": {},
  "evidence_scope": "same pretrained parent; one pilot plus two fixed repeats; known exposure audit only",
  "legacy_test_role": "legacy_exposed_regression",
  "not_innovation_proof": true
}
```

全部逐图记录、场景bootstrap、固定样例和后验失败面板在final/<seed>/<role>/中。
