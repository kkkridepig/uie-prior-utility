# Data and exposure audit

| Role | Groups | Pairs |
| --- | --- | --- |
| utility_fit | 440 | 443 |
| utility_val | 132 | 136 |
| calibration | 132 | 133 |
| sealed_eval | 176 | 177 |
| model_fit | 3330 | 3608 |
| model_val | 588 | 671 |

Full local LSUI 4279/UIEB 890 input/reference audit; 4 cross-source edges, one UIEB pair excluded; 1767 near edges and 103 exact edges.
Grouping is content_group_proxy, not verified semantic scenes. All samples are historically_analyzed.
The official source declares LSUI; per-image author training membership remains unknown. Local LSUI is an exposed candidate development pool, not an author-split reproduction.
UIEB documented_nonoverlap is scoped to the declared LSUI source and the available full-file cross-audit. This does not prove universal nonoverlap or a new blind test.
RoleGuard rejects independent labels when provenance is unknown and sealed references until DEV_PASS plus freeze.
Lists and hashes: all_pairs.jsonl, roles.jsonl, groups.json, exposure_ledger.json, near_duplicate_edges.jsonl, split_freeze.json.
