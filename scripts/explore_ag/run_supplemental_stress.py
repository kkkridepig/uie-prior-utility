"""Run fixed source stress diagnostics after the existing device queue and A retry."""
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explore_ag import dispatch


def main():
    with (dispatch.RUN/'supplemental_stress.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while dispatch.load(dispatch.RUN/'control_recovery.json', {}).get('status') not in ('completed','failed','not_needed'):
            time.sleep(30)
        while True:
            try:
                runner = dispatch.Dispatcher()
                break
            except BlockingIOError:
                time.sleep(30)
        original_status = runner.state['status']
        tasks = [('P1','PARENT',dispatch.RUN/'parent_0.pt')]
        for package, branches in dispatch.GROUPS:
            for branch in branches:
                checkpoint = dispatch.RUN/'branches'/branch/'step_05000.pt'
                if checkpoint.exists() and runner.state['branches'].get(branch,{}).get('status')=='completed_screen':
                    tasks.append((package,branch,checkpoint))
        solver,nfe = runner.state['eval_protocol'].split(':')
        coverage = {}
        for package, branch, checkpoint in tasks:
            output = dispatch.RUN/'supplemental_stress'/branch
            command = [dispatch.PYTHON,'-u','scripts/explore_ag/stress.py',
                       '--checkpoint',str(checkpoint),'--output',str(output),
                       '--solver',solver,'--steps',nfe,
                       '--deadline-unix',str(runner.deadline(package))]
            ok = runner.complete_resumable('STRESS_'+branch, package, command,
                                          output/'summary.json', 1800)
            coverage[branch] = 'completed' if ok else 'incomplete_or_blocked'
            dispatch.atomic_json(dispatch.RUN/'supplemental_stress_status.json',
                                 {'status':'running','coverage':coverage})
        runner.state['status'] = original_status
        runner.save()
        dispatch.atomic_json(dispatch.RUN/'supplemental_stress_status.json',
                             {'status':'completed' if all(x=='completed' for x in coverage.values()) else 'partial',
                              'coverage':coverage,'test_selection_changed':False})
        lines = ['# 补充先验扰动诊断', '',
                 '仅使用固定 24 张源验证图；该补充协议在初轮干净验证之后固定，不能称作训练前预注册。',
                 '不改变已冻结的测试方案，不据此反复优化测试成绩。各训练分支取共同新增 5000 步。', '',
                 '| 分支 | 范围 | 扰动 | 图数 | 干净减扰动 PSNR (dB) |',
                 '|---|---|---|---:|---:|']
        for _,branch,_ in tasks:
            summary = dispatch.load(dispatch.RUN/'supplemental_stress'/branch/'summary.json',{})
            for scope,cases in summary.items():
                for case,value in cases.items():
                    lines.append('| {} | {} | {} | {} | {:.6f} |'.format(
                        branch,scope,case,value['count'],value['mean_drop_db']))
        lines += ['', 'all_priors 重新计算物理反演条件；added_route_only 只替换 C 新增残差的输入，父通路保持干净。',
                  '降幅小必须结合干净绝对质量解释。逐图 PSNR/SSIM、协议与缺项：`runs/explore_ag_single_seed_v2_20261003/supplemental_stress/`。']
        (dispatch.DOC/'AG_V2_SUPPLEMENTAL_STRESS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
