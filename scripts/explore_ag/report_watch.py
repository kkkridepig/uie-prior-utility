"""Read-only, resumable Markdown reporting for the long-running A-G dispatcher."""
import argparse
import json
from pathlib import Path
import time

from scripts.explore_ag import dispatch
from scripts.explore_ag.summarize import paired_delta, read_records


ROOT = dispatch.RUN
DOC = dispatch.DOC
CONTROLS = {
    'B_PROXY': 'BASE_CONT', 'A_COUPLED': 'BASE_CONT',
    'A_SPLIT': 'A_COUPLED', 'C_CONV': 'BASE_CONT',
    'C_FIXED': 'C_CONV', 'C_ROUTED': 'C_FIXED',
    'D_SOBEL_CONTROL': 'BASE_CONT', 'D_PHASE': 'D_SOBEL_CONTROL',
    'F_UNIFORM': 'BASE_CONT', 'F_ROUTED': 'F_UNIFORM',
}
TERMINAL = {'screening_finished', 'screening_finished_with_documented_blocks',
            'stopped_error', 'waiting_for_baseline'}


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_if_changed(path, content):
    if path.exists() and path.read_text() == content:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def duration_by_id(ledger):
    result = {}
    for event in ledger['events']:
        if event.get('event') == 'task_end':
            result[event['id']] = result.get(event['id'], 0) + event['elapsed_seconds']
    return result


def task_seconds(ledger, prefix):
    return sum(seconds for name, seconds in duration_by_id(ledger).items()
               if name.startswith(prefix))


def render_e0(run, doc, ledger):
    full = load(run/'e0/full_latest/summary.json')
    subset = load(run/'e0/subset_latest/summary.json')
    if not subset:
        return
    lines = ['# E0: source-validation sampler screening', '',
             'Same latest recoverable checkpoint, sample-keyed noise and UIEB grouped_v1 validation. '
             'This is not test-set evidence. Each reported NFE is the actual denoiser-call count.', '',
             '| Split | Sampler | Images | PSNR (dB) | SSIM | Median end-to-end (s) |',
             '|---|---|---:|---:|---:|---:|']
    for label, table in (('fixed 24', subset), ('full 91', full)):
        if not table:
            continue
        for solver in ('ddpm:1000', 'ddim:20', 'ddim:50', 'ddim:100'):
            row = table.get(solver)
            if row:
                lines.append('| {} | {} | {} | {:.6f} | {:.6f} | {:.3f} |'.format(
                    label, solver, row['count'], row['mean_psnr'], row['mean_ssim'],
                    row['median_end_to_end_seconds']))
    lines += ['', 'E package charged device time so far: {:.3f} h. '
              'Full validation and parent confirmation must pass before the new sampler is frozen.'.format(
                  sum(e.get('elapsed_seconds', 0) for e in ledger['events']
                      if e.get('event') == 'task_end' and e.get('package') == 'E')/3600),
              'Protocol: `runs/explore_ag_single_seed_v2_20261003/e0/eval_protocol_v2.json` '
              '(exists only after full validation); per-image records: `e0/full_latest/per_image.jsonl`.', '']
    write_if_changed(doc/'AG_V2_E0_RESULTS.md', '\n'.join(lines))


def render_branch(run, doc, branch, state, ledger):
    directory = run/'branches'/branch
    config = load(directory/'run_config.json')
    if config is None:
        return
    lines = ['# {}: single-seed screening'.format(branch), '',
             'Evidence state: `{}`; added optimizer steps: {}. Source validation only. '
             'All trained branches start independently from the same frozen parent.'.format(
                 state.get('status', 'running'), state.get('added_steps', 0)), '',
             'Parent SHA-256: `{}`; parent training step: {}. '
             'Train/evaluate implementation SHA-256: `{}`.'.format(
                 config['parent_checkpoint_sha256'], config['parent_training_steps'],
                 config['implementation_sha256']), '',
             'Trainable parameter groups: `{}`. Migration: `{}`.'.format(
                 config['parameter_groups'], config['migration']), '',
             'Effective global batch 4 = micro batch {} x accumulation {}; '
             'new parameters and inherited parameters use the recorded separate learning rates.'.format(
                 config['micro_batch'], config['gradient_accumulation']), '',
             'Recorded device occupation for this branch: {:.3f} h '
             '(completed dispatcher tasks only; the active task is charged on completion).'.format(
                 task_seconds(ledger, branch+'_')/3600), '',
             '| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |',
             '|---:|---:|---:|---:|---|---:|---|']
    for step in (1000, 2000, 3000, 4000, 5000, 10000):
        val = directory/('val_{:05d}'.format(step))
        summary = load(val/'summary.json')
        if not summary:
            continue
        control = CONTROLS.get(branch)
        delta = None
        if control:
            control_val = run/'branches'/control/('val_{:05d}'.format(step))
            if load(control_val/'summary.json'):
                delta = paired_delta(read_records(val/'per_image.jsonl'),
                                     read_records(control_val/'per_image.jsonl'), 'psnr')
        lines.append('| {} | {} | {:.6f} | {:.6f} | {} | {} | {} |'.format(
            step, summary['count'], summary['mean_psnr'], summary['mean_ssim'],
            control or 'PARENT_0',
            '{:+.6f}'.format(delta['mean_delta']) if delta and delta['mean_delta'] is not None else 'pending',
            '[{:+.4f}, {:+.4f}]'.format(*delta['ci95']) if delta and delta['ci95'] else 'pending'))
    lines += ['', 'Matched comparison uses the same added-step checkpoint and source validation images. '
              'The interval resamples fixed-weight validation scenes/images, not training seeds. '
              'Any difference also incurs the added training cost shown above.',
              'Checkpoints, train log, per-image metrics and worst cases are in '
              '`runs/explore_ag_single_seed_v2_20261003/branches/{}/`.'.format(branch), '']
    write_if_changed(doc/('AG_V2_{}_RESULTS.md'.format(branch)), '\n'.join(lines))


def render_progress(run=ROOT, doc=DOC):
    ledger = load(run/'budget.json')
    state = load(run/'dispatch_state.json')
    if ledger is None or state is None:
        raise FileNotFoundError('Dispatcher budget or state file missing')
    render_e0(run, doc, ledger)
    for _, branches in dispatch.GROUPS:
        for branch in branches:
            render_branch(run, doc, branch, state.get('branches', {}).get(branch, {}), ledger)
    ended = ledger.get('legacy_queue_device_seconds_since_start', 0) + ledger.get('new_experiment_device_seconds', 0)
    open_events = {}
    for event in ledger['events']:
        if event.get('event') == 'task_start':
            open_events[event['id']] = event
        elif event.get('event') == 'task_end':
            open_events.pop(event['id'], None)
    active = sum(max(0, time.time()-event['unix']) for event in open_events.values())
    lines = ['# A-G live exploration status', '',
             'This file reflects dispatcher records, not a completed seven-day experiment. '
             'The authoritative budget remains `runs/explore_ag_single_seed_v2_20261003/budget.json`.', '',
             'Dispatcher status: `{}`; active task: `{}`.'.format(
                 state.get('status'), state.get('current_task')), '',
             'Charged device occupation: {:.3f} h completed + {:.3f} h active = {:.3f} h / 168 h. '
             'Exploration ceiling: 152 h; final Benchmark reserve: 16 h.'.format(
                 ended/3600, active/3600, (ended+active)/3600), '',
             'Parent: `{}`; evaluation protocol: `{}`.'.format(
                 state.get('parent', {}).get('sha256', 'pending'),
                 state.get('eval_protocol', 'pending')), '',
             '| Direction/branch | State | Added steps | Full source validation |',
             '|---|---|---:|---|']
    branches = [name for _, group in dispatch.GROUPS for name in group]
    branches += ['B_ADAPT', 'G_FEATURE', 'G0_G1']
    for branch in branches:
        item = state.get('branches', {}).get(branch, {})
        last = item.get('latest_val', {})
        score = '{:.4f} dB / {:.5f} SSIM'.format(last['mean_psnr'], last['mean_ssim']) if last else 'pending'
        lines.append('| {} | {} | {} | {} |'.format(
            branch, item.get('status', 'planned'), item.get('added_steps', 0), score))
    lines += ['', 'E0 and each branch with a run configuration have separate `AG_V2_*_RESULTS.md` '
              'records in this directory. Missing results remain pending or blocked; no test score is '
              'used for model selection.', '']
    recovery = load(run/'control_recovery.json', {})
    stress = load(run/'supplemental_stress_status.json', {})
    lines += ['Supplemental A control recovery: `{}`; source-only stress diagnostics: `{}`. '
              'These queues share the device lock and cumulative ledger; their existence does not '
              'mean the diagnostics have completed.'.format(recovery.get('status', 'queued'),
                                                          stress.get('status', 'queued')), '']
    write_if_changed(doc/'AG_V2_LIVE_STATUS.md', '\n'.join(lines))
    if state.get('status') in TERMINAL and (run/'selection_freeze_before_test.json').exists():
        from scripts.explore_ag.summarize import summarize
        result = summarize(run)
        destination = run/'benchmark'/'paired_summary.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(destination, json.dumps(result, indent=2)+'\n')
    if state.get('status') in TERMINAL:
        render_final(run, doc, state, ledger)
    return state.get('status')


def render_final(run, doc, state, ledger):
    charged = (ledger.get('legacy_queue_device_seconds_since_start', 0)
               + ledger.get('new_experiment_device_seconds', 0)) / 3600
    lines = ['# A-G single-seed evidence at dispatcher stop', '',
             'Terminal state: `{}`. Charged single-device time: {:.3f} / 168 h; '
             'the final Benchmark has a separate 16 h reserve.'.format(state['status'], charged), '',
             'This is one shared-parent exploratory seed. An implemented branch, a passed CPU test, '
             'a PPU training run, and a supported improvement are distinct evidence levels.', '',
             '| Branch | Status | Added steps | 5000-step validation PSNR | Limitation |',
             '|---|---|---:|---:|---|']
    names = [name for _, group in dispatch.GROUPS for name in group]
    names += ['B_ADAPT', 'G_FEATURE', 'G0_G1']
    completed = 0
    for name in names:
        entry = state.get('branches', {}).get(name, {})
        if entry.get('status') == 'completed_screen':
            completed += 1
        score = load(run/'branches'/name/'val_05000'/'summary.json')
        lines.append('| {} | {} | {} | {} | {} |'.format(
            name, entry.get('status', 'not_run'), entry.get('added_steps', 0),
            '{:.4f}'.format(score['mean_psnr']) if score else 'not measured',
            entry.get('reason') or entry.get('limitation') or ''))
    lines += ['', 'Completed training screens: {}/{} planned branch/control runs. '
              'E0 is separate and requires no added training.'.format(completed, len(names)), '',
              'Selected parent: `{}`; fixed sampler: `{}`. '
              'Selection used UIEB validation only.'.format(
                  state.get('parent', {}).get('sha256', 'not selected'),
                  state.get('eval_protocol', 'not frozen')), '',
              'Per-branch training, migration, matched-step validation deltas, and cost: '
              '`docs/experiments/AG_V2_*_RESULTS.md`. Raw checkpoints, per-image records, '
              'logs and the cumulative ledger: `runs/explore_ag_single_seed_v2_20261003/`.', '']
    coverage = state.get('benchmark_coverage', {})
    lines += ['## Frozen Benchmark', '',
              'Frozen test selection: `{}`. '
              'Test and cross-dataset numbers below are never used to change the selected model.'.format(
                  'yes' if (run/'selection_freeze_before_test.json').exists() else 'no'), '',
              '| Model/dataset | Coverage | Images | PSNR (dB) | SSIM | Median end-to-end (s) |',
              '|---|---|---:|---:|---:|---:|']
    for key, status in sorted(coverage.items()):
        branch, dataset = key.split('/', 1)
        result = load(run/'benchmark'/branch/dataset/'summary.json')
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            key, status, result['count'] if result else 'not measured',
            '{:.4f}'.format(result['mean_psnr']) if result and result['mean_psnr'] is not None else 'null',
            '{:.5f}'.format(result['mean_ssim']) if result and result['mean_ssim'] is not None else 'null',
            '{:.3f}'.format(result['median_seconds']) if result else 'null'))
    if not coverage:
        lines.append('| none | not_run | null | null | null | null |')
    lines += ['', 'Paired per-image differences, improvement fractions, 1000-draw intervals, and '
              'the UIEB 96-image scene-overlap sensitivity are in '
              '`runs/explore_ag_single_seed_v2_20261003/benchmark/paired_summary.json` when available.', '',
              'Atlantis is a licensed synthetic relative-depth candidate, not measured underwater '
              'geometry. RUOD dataset permission and an audited underwater detector remain unresolved; '
              'no G AP is reported without a real backend and annotations. LPIPS and audited '
              'no-reference metrics remain null if their weights/formulas are unavailable. '
              'A single training seed cannot establish initialization stability.', '']
    write_if_changed(doc/'AG_V2_FINAL_EVIDENCE.md', '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=300)
    args = parser.parse_args()
    while True:
        status = render_progress()
        print(json.dumps({'status': status, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}), flush=True)
        if args.once or status in TERMINAL:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
