# 本轮实际执行命令与复查入口

工作目录为 `/mnt/workspace/uie-prior-utility`。下面的恢复、下载及测试命令均实际执行过；未运行正式 train 或 pipeline。回执均写入 `runs/server_recovery_20261007/`，避免与历史记录混淆。

## 环境与历史记录

```bash
git clone https://github.com/kkkridepig/uie-prior-utility.git /mnt/workspace/uie-prior-utility
cd /mnt/workspace/uie-prior-utility
python3 tools/restore_recovery_records.py
python3 tools/restore_recovery_records.py --apply
python3 -m venv --system-site-packages .venv
bash scripts/bootstrap_v3_recovery_ppu.sh
```

依赖安装曾分批完成；最后使用上述 bootstrap 重查全部最终版本并通过 PPU doctor。`requirements-recovery-ppu.txt` 是本轮最终依赖约束，禁止用旧 bootstrap 替换厂商 torch。

## UIEB 与公开依赖

RARLAB 工具下载到 `downloads/rar/` 并展开为 `downloads/rar/rar/unrar`。UIEB 完整性测试与解压已使用用户提供的密码执行。以下是等价的复查命令，使用交互读取密码的方式，避免密码再次写入文档；交互读取这一行本身不是本轮原执行记录。

```bash
read -r -s -p 'RAR password: ' UIE_ARCHIVE_PASSWORD
downloads/rar/rar/unrar t -p"$UIE_ARCHIVE_PASSWORD" -idq /mnt/workspace/TEMP-FILE-STATION/raw-890-s.rar
downloads/rar/rar/unrar t -p"$UIE_ARCHIVE_PASSWORD" -idq /mnt/workspace/TEMP-FILE-STATION/reference-890.rar
downloads/rar/rar/unrar e -o- -p"$UIE_ARCHIVE_PASSWORD" -idq /mnt/workspace/TEMP-FILE-STATION/raw-890-s.rar data/uieb/raw-890/
downloads/rar/rar/unrar e -o- -p"$UIE_ARCHIVE_PASSWORD" -idq /mnt/workspace/TEMP-FILE-STATION/reference-890.rar data/uieb/reference-890/
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git -C third_party/Depth-Anything-V2 checkout --detach a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
python3 scripts/fetch_mpa_weights.py
```

另外，实际通过同一 `scripts.fetch_mpa_weights.fetch` 函数下载 VGG16：

```bash
python3 - <<'PY'
from scripts.fetch_mpa_weights import fetch
fetch('cde_v3/vgg16-397923af.pth', (
    'https://download.pytorch.org/models/vgg16-397923af.pth',
    553433881,
    '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'))
PY
```

下载函数没有完整文件的跳过机制；已有权重无需重复下载。所有下载 receipt 与完整哈希已保存。

## 合成测试数据与验证

恢复合成 fixture 的实际调用如下。生成输出清单与原清单哈希一致，原清单没有改写。已有 fixture 无需重复生成。

```bash
.venv/bin/python - <<'PY'
from mpa_diff.data.synthetic import generate
generate('data/synthetic_fixture', 'runs/server_recovery_20261007/synthetic_fixture_generated.jsonl')
PY
.venv/bin/python -m pytest --junitxml=runs/server_recovery_20261007/tests_all_after_compatibility.xml
.venv/bin/python tools/recovery_ppu_smoke.py --device cpu --output runs/server_recovery_20261007/v3_cpu_smoke.json
.venv/bin/python tools/recovery_ppu_smoke.py --device cuda --output runs/server_recovery_20261007/v3_ppu_smoke.json
.venv/bin/python tools/recovery_public_dependencies_smoke.py --device cuda --output runs/server_recovery_20261007/public_ppu/dependency_smoke.json
.venv/bin/python -m compileall -q uie mpa_diff scripts/cde_v3 tools
git diff --check
python3 tools/restore_recovery_records.py
```

冒烟回执有拒绝覆盖保护；以上输出已存在，重查时必须使用新文件名，公开依赖检查须用新的父目录。两个工具均没有 optimizer 更新，不会创建研究模型权重。

数据逐文件审计实际调用了原 `mpa_diff.data.manifest.audit`，输出保存在 `data_audit.json`；LSUI 审计失败与 ZIP 全文件扫描结果原样保留。LSUI 未解压为正式数据，未运行新随机数据划分。

## 日常复查

```bash
cd /mnt/workspace/uie-prior-utility
source .venv/bin/activate
python -m uie doctor --device cuda --precision fp32
python -m pytest
python tools/restore_recovery_records.py
```

这里只提供测试入口。没有原父权重之前，不运行 `scripts/cde_v3/pipeline.py`、旧 run_pipeline 或新的 400000 步训练来冒充恢复。
