"""Persistent, budgeted single-device exploration dispatcher."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.utils.io import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/explore_ag_single_seed_v2_20261003'
DOC = ROOT/'docs/experiments'
PYTHON = str(ROOT/'.venv/bin/python')
LEGACY = ROOT/'runs/s2_grouped_v1/uieb/seed_20260927'
PREVIOUS = RUN/'legacy_step_220000.pt'
LIMITS = {'P0': 8, 'E': 8, 'P1': 16, 'B': 12, 'A': 12, 'C': 20,
          'D': 12, 'F': 12, 'G': 24, 'FINAL': 16}
GROUPS = [('P1', ['BASE_CONT']), ('B', ['B_PROXY']),
          ('A', ['A_COUPLED', 'A_SPLIT']),
          ('C', ['C_CONV', 'C_FIXED', 'C_ROUTED']),
          ('D', ['D_SOBEL_CONTROL', 'D_PHASE']),
          ('F', ['F_UNIFORM', 'F_ROUTED'])]


def atomic_json(path, data):
    temp = Path(str(path)+'.tmp')
    write_json(temp, data)
    temp.replace(path)


def load(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Dispatcher:
    def __init__(self):
        RUN.mkdir(parents=True, exist_ok=True)
        self.lock = (RUN/'device.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.ledger_path = RUN/'budget.json'
        self.ledger = load(self.ledger_path, None)
        if self.ledger is None:
            raise RuntimeError('Original cumulative budget missing; refusing to reset')
        self.state_path = RUN/'dispatch_state.json'
        self.state = load(self.state_path, {'status': 'waiting_for_safe_legacy_checkpoint',
                                            'branches': {}, 'tasks': [], 'started_utc': timestamp()})
        self.reconcile_unclosed_tasks()

    def reconcile_unclosed_tasks(self):
        pending = {}
        for event in self.ledger['events']:
            if event.get('event') == 'task_start': pending[event['id']] = event
            elif event.get('event') == 'task_end': pending.pop(event['id'], None)
        for task_id, event in pending.items():
            pid = event.get('pid')
            path = Path('/proc/{}'.format(pid)) if pid else None
            while path and path.exists():
                try:
                    cmdline = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()
                except OSError:
                    break
                if 'scripts/explore_ag/' not in cmdline:
                    break
                self.state['status'] = 'waiting_for_previous_task_to_finish'
                self.save()
                time.sleep(30)
            log_path = RUN/'logs'/('{}.log'.format(task_id.replace('/','_')))
            end = max(event['unix'], log_path.stat().st_mtime) if log_path.exists() else time.time()
            elapsed = end-event['unix']
            self.ledger['new_experiment_device_seconds'] += elapsed
            self.ledger['events'].append({'event':'task_end', 'id':task_id,
                                          'package':event['package'], 'utc':timestamp(),
                                          'elapsed_seconds':elapsed, 'exit_code':None,
                                          'recovered_after_dispatcher_restart':True,
                                          'log':str(log_path)})
            self.state['tasks'].append({'id':task_id, 'status':'interrupted_or_unknown',
                                        'elapsed_seconds':elapsed, 'log':str(log_path)})
            self.save()

    def save(self):
        atomic_json(self.ledger_path, self.ledger)
        atomic_json(self.state_path, self.state)

    def charged_seconds(self):
        legacy = self.ledger.get('legacy_queue_device_seconds_since_start') or 0
        return legacy + self.ledger.get('new_experiment_device_seconds', 0)

    def package_seconds(self, package):
        charged = sum(e.get('elapsed_seconds', 0) for e in self.ledger['events']
                      if e.get('event') == 'task_end' and e.get('package') == package)
        if package == 'P0':
            charged += self.ledger.get('legacy_queue_device_seconds_since_start') or 0
        return charged

    def deadline(self, package):
        global_limit = 168 if package == 'FINAL' else 152
        global_left = global_limit*3600 - self.charged_seconds()
        borrowed_elsewhere = sum(max(0, self.package_seconds(p) - limit*3600)
                                 for p, limit in LIMITS.items() if p != package)
        common_left = max(0, 28*3600 - borrowed_elsewhere)
        package_left = LIMITS[package]*3600 + common_left - self.package_seconds(package)
        return time.time() + max(0, min(global_left, package_left))

    def can_start(self, package, estimate_seconds=1200):
        return self.deadline(package) - time.time() >= estimate_seconds + 900

    def run(self, task_id, package, command, estimate_seconds=1200):
        log_path = RUN/'logs'/('{}{}.log'.format(task_id.replace('/','_'), ''))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.can_start(package, estimate_seconds):
            self.state['tasks'].append({'id': task_id, 'status': 'budget_insufficient', 'at': timestamp()})
            self.save()
            return False
        deadline = self.deadline(package)
        if '--deadline-unix' not in command and any(x in ' '.join(command) for x in ('train.py', 'evaluate.py', 'benchmark.py', 'run_e0.py', 'profile.py')):
            command += ['--deadline-unix', str(deadline)]
        begin = time.time()
        event = {'event': 'task_start', 'id': task_id, 'package': package,
                 'utc': timestamp(), 'unix': begin, 'command': command}
        self.ledger['events'].append(event)
        self.state['current_task'] = task_id
        self.state['status'] = 'running'
        self.save()
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            event['pid'] = process.pid
            self.save()
            code = process.wait()
        elapsed = time.time()-begin
        self.ledger['new_experiment_device_seconds'] += elapsed
        self.ledger['events'].append({'event': 'task_end', 'id': task_id, 'package': package,
                                      'utc': timestamp(), 'elapsed_seconds': elapsed,
                                      'exit_code': code, 'log': str(log_path)})
        self.state['tasks'].append({'id': task_id, 'status': 'completed' if code == 0 else 'failed_implementation',
                                    'elapsed_seconds': elapsed, 'log': str(log_path)})
        self.state['current_task'] = None
        self.save()
        return code == 0

    def complete_resumable(self, task_id, package, command, summary_path, estimate_seconds):
        """Resume partial evaluations while their package and global budgets allow."""
        while not summary_path.exists():
            progress = summary_path.parent/'per_image.jsonl'
            before = progress.stat().st_size if progress.exists() else 0
            if not self.run(task_id, package, list(command), estimate_seconds):
                return False
            if summary_path.exists():
                return True
            after = progress.stat().st_size if progress.exists() else 0
            if after <= before:
                self.state['tasks'].append({'id': task_id, 'status': 'incomplete_no_progress',
                                            'at': timestamp()})
                self.save()
                return False
        return True

    def wait_for_legacy(self):
        receipt_path = RUN/'old_queue_park_receipt.json'
        while not receipt_path.exists():
            if not Path('/proc/1423731').exists():
                raise RuntimeError('Old queue exited without verified checkpoint receipt')
            self.state['status'] = 'waiting_for_safe_legacy_checkpoint'
            self.save()
            time.sleep(30)
        receipt = json.loads(receipt_path.read_text())
        if receipt['step'] < 225000 or receipt['sha256'] != sha256(LEGACY/'last.pt'):
            raise ValueError('Parked checkpoint receipt does not match latest file')
        for _ in range(30):
            if not Path('/proc/1423731').exists(): break
            time.sleep(1)
        if Path('/proc/1423731').exists():
            raise RuntimeError('Old queue still alive after parking receipt')
        import calendar
        start = calendar.timegm(time.strptime(self.ledger['started_utc'], '%Y-%m-%dT%H:%M:%SZ'))
        elapsed = max(0, receipt['verified_unix'] - start)
        if self.ledger['legacy_queue_device_seconds_since_start'] is None:
            self.ledger['legacy_queue_device_seconds_since_start'] = elapsed
            self.ledger['events'].append({'event': 'legacy_queue_parked', 'package': 'P0',
                                          'utc': timestamp(), 'elapsed_seconds': elapsed,
                                          'checkpoint_step': receipt['step'],
                                          'checkpoint_sha256': receipt['sha256']})
        self.state['legacy_park_receipt'] = str(receipt_path)
        self.save()

    def run_e0(self):
        latest = LEGACY/'last.pt'
        out = RUN/'e0'
        subset = out/'subset_latest'
        full = out/'full_latest'
        if not (subset/'summary.json').exists():
            command = [PYTHON, '-u', 'scripts/explore_ag/run_e0.py',
                       '--checkpoint', str(latest), '--output', str(subset)]
            if not self.complete_resumable('E0_subset', 'E', command, subset/'summary.json', 1800):
                raise RuntimeError('E0 subset incomplete; resumable records retained')
        preliminary = json.loads((subset/'summary.json').read_text())
        candidates = sorted((k for k in preliminary if k.startswith('ddim:')),
                            key=lambda k: int(k.split(':')[1]))
        solvers = ['ddpm:1000'] + candidates
        if not (full/'summary.json').exists():
            command = [PYTHON, '-u', 'scripts/explore_ag/run_e0.py',
                       '--checkpoint', str(latest), '--output', str(full), '--full',
                       '--solvers'] + solvers
            if not self.complete_resumable('E0_full', 'E', command, full/'summary.json', 3600):
                raise RuntimeError('E0 full validation incomplete; resumable records retained')
        results = json.loads((full/'summary.json').read_text())
        ddpm = results['ddpm:1000']
        options = [k for k in candidates if results[k]['mean_psnr'] >= ddpm['mean_psnr']-.10
                   and results[k]['mean_ssim'] >= ddpm['mean_ssim']-.002]
        chosen = options[0] if options else 'ddpm:1000'
        solver, nfe = chosen.split(':')
        atomic_json(out/'eval_protocol_v2.json', {'id': 'eval_protocol_v2', 'solver': solver,
                    'nfe': int(nfe), 'eta': 0., 'noise': 'sample-keyed SHA256',
                    'chosen_on': 'UIEB grouped_v1 full validation',
                    'source_checkpoint_sha256': sha256(latest), 'threshold': {'psnr_db': .10, 'ssim': .002}})
        self.state['eval_protocol'] = chosen
        self.save()
        return solver, int(nfe), ddpm, results[chosen]

    def select_parent(self, solver, nfe, latest_result):
        latest = LEGACY/'last.pt'
        best = LEGACY/'best.pt'
        candidates = [('latest', latest), ('best', best)]
        if PREVIOUS.exists():
            candidates.append(('previous', PREVIOUS))
        unique = {}
        for label, path in candidates:
            unique.setdefault(sha256(path), (label, path))

        def score_candidate(label, path, chosen_solver, chosen_nfe):
            if path == latest:
                return json.loads((RUN/'e0/full_latest/summary.json').read_text())['{}:{}'.format(chosen_solver, chosen_nfe)]
            output = RUN/'parent_candidates'/('{}_{}_{}'.format(label, chosen_solver, chosen_nfe))
            if not self.complete_resumable('P1_{}_{}_{}'.format(label, chosen_solver, chosen_nfe),
                                           'P1', [PYTHON, '-u', 'scripts/explore_ag/evaluate.py',
                                           '--checkpoint', str(path), '--output', str(output),
                                           '--solver', chosen_solver, '--steps', str(chosen_nfe)],
                                           output/'summary.json', 5400 if chosen_solver == 'ddpm' else 1200):
                raise RuntimeError('Parent candidate source validation failed: {}'.format(label))
            return json.loads((output/'summary.json').read_text())

        options = [(path, score_candidate(label, path, solver, nfe))
                   for label, path in unique.values()]
        # Earlier step wins a practical tie smaller than 0.01 dB.
        high = max(score['mean_psnr'] for _,score in options)
        eligible = [(path,score) for path,score in options if high-score['mean_psnr'] < .01]
        selected, score = min(eligible, key=lambda pair: torch_step(pair[0]))
        if selected != latest and solver == 'ddim':
            confirm = RUN/'parent_candidates'/('{}_sampler_confirmation'.format(selected.stem))
            labels = ['ddpm:1000', 'ddim:{}'.format(nfe)]
            if not (confirm/'summary.json').exists():
                command = [PYTHON, '-u', 'scripts/explore_ag/run_e0.py',
                           '--checkpoint', str(selected), '--output', str(confirm),
                           '--full', '--solvers'] + labels
                if not self.complete_resumable('P1_best_sampler_confirmation', 'P1', command,
                                               confirm/'summary.json', 3600):
                    raise RuntimeError('Selected-parent sampler confirmation incomplete')
            confirmed = json.loads((confirm/'summary.json').read_text())
            reference = confirmed['ddpm:1000']
            fast = confirmed['ddim:{}'.format(nfe)]
            if fast['mean_psnr'] < reference['mean_psnr']-.10 or fast['mean_ssim'] < reference['mean_ssim']-.002:
                # This is the one permitted protocol adjustment. Recompare both
                # parent candidates under DDPM before any branch training.
                solver, nfe = 'ddpm', 1000
                options = [(path, reference if path == selected else score_candidate(label, path, solver, nfe))
                           for label, path in unique.values()]
                high = max(s['mean_psnr'] for _, s in options)
                selected, score = min(((p, s) for p, s in options if high-s['mean_psnr'] < .01),
                                      key=lambda pair: torch_step(pair[0]))
                self.state['eval_protocol'] = 'ddpm:1000'
                protocol_path = RUN/'e0/eval_protocol_v2.json'
                protocol = load(protocol_path, {})
                protocol.update(solver='ddpm', nfe=1000,
                                adjustment='Best-parent DDIM confirmation failed quality threshold',
                                confirmed_parent_sha256=sha256(selected))
                atomic_json(protocol_path, protocol)
        frozen = RUN/'parent_0.pt'
        selected_sha = sha256(selected)
        if frozen.exists():
            if sha256(frozen) != selected_sha:
                raise ValueError('Frozen parent already exists with another hash')
        else:
            os.link(selected, frozen)
        receipt = {'selected_checkpoint': str(frozen.resolve()), 'original_checkpoint': str(selected.resolve()),
                   'sha256': selected_sha,
                   'training_steps': torch_step(selected), 'mean_psnr': score['mean_psnr'],
                   'mean_ssim': score['mean_ssim'], 'candidates': [
                       {'path': str(p.resolve()), 'sha256': sha256(p), 'step': torch_step(p),
                        'mean_psnr': s['mean_psnr'], 'mean_ssim': s['mean_ssim']} for p,s in options],
                   'selection_split': 'UIEB grouped_v1 validation', 'sampler': self.state['eval_protocol'],
                   'previous_complete_checkpoint': str(PREVIOUS) if PREVIOUS.exists() else 'unavailable'}
        prior_receipt = RUN/'parent_selection.json'
        if prior_receipt.exists() and load(prior_receipt, {})['sha256'] != selected_sha:
            raise ValueError('Previously selected parent differs; refusing to reselect')
        atomic_json(prior_receipt, receipt)
        self.state['parent'] = receipt
        protocol_state_path = RUN/'protocol_state.json'
        protocol_state = load(protocol_state_path, {})
        protocol_state.update(parent_checkpoint_sha256=receipt['sha256'],
                              parent_training_steps=receipt['training_steps'],
                              eval_protocol_id='eval_protocol_v2',
                              evidence_status='running', stage='P1_parent_selected')
        atomic_json(protocol_state_path, protocol_state)
        self.save()
        return frozen, solver, nfe

    def train_groups(self, parent, solver, nfe, micro_batch):
        profile = load(RUN/'performance/summary.json', {})
        speed = profile.get('results', {}).get(str(micro_batch), {}).get('mean_update_seconds')
        sampler = load(RUN/'e0/full_latest/summary.json', {}).get('{}:{}'.format(solver, nfe), {})
        per_image = sampler.get('median_end_to_end_seconds')
        if not speed or not per_image:
            raise RuntimeError('Measured update and validation costs are required before branch dispatch')
        for package, branches in GROUPS:
            for target in (1000,2000,3000,4000,5000):
                if target in (1000, 3000):
                    milestone = 2000 if target == 1000 else 5000
                    needed = 900
                    for branch in branches:
                        directory = RUN/'branches'/branch
                        current = checkpoint_step(directory/'last.pt')
                        needed += max(0, milestone-current)*speed*1.2
                        first = 1000 if milestone == 2000 else 3000
                        for check_step in range(first, milestone+1, 1000):
                            if not (directory/('val_{:05d}'.format(check_step))/'summary.json').exists():
                                needed += (91 if check_step in (2000, 5000) else 24)*per_image*1.2
                    if self.deadline(package)-time.time() < needed:
                        for branch in branches:
                            previous = self.state['branches'].get(branch, {})
                            if previous.get('status') != 'completed_screen':
                                self.state['branches'][branch] = {
                                    'status': 'budget_insufficient',
                                    'added_steps': checkpoint_step(RUN/'branches'/branch/'last.pt'),
                                    'reason': 'Matched group cannot reach and validate {} steps in remaining package/global budget'.format(milestone),
                                    'estimated_group_seconds': needed}
                        self.save()
                        self.write_direction(package, branches)
                        break
                for branch in branches:
                    if self.state['branches'].get(branch,{}).get('status') in ('failed_implementation','budget_insufficient'):
                        continue
                    directory = RUN/'branches'/branch
                    current = checkpoint_step(directory/'last.pt')
                    if current < target:
                        command = [PYTHON, '-u', 'scripts/explore_ag/train.py', '--parent', str(parent),
                                   '--branch', branch, '--output', str(directory),
                                   '--target-steps', str(target), '--micro-batch', str(micro_batch)]
                        if not self.run(branch+'_train_'+str(target), package, command,
                                        max(1200, (target-current)*1.5)):
                            self.state['branches'][branch] = {'status': 'failed_implementation' if self.can_start(package) else 'budget_insufficient',
                                                               'added_steps': checkpoint_step(directory/'last.pt')}
                            self.save()
                            continue
                    current = checkpoint_step(directory/'last.pt')
                    if current < target:
                        self.state['branches'][branch] = {'status': 'budget_insufficient', 'added_steps': current}
                        self.save()
                        continue
                    check = directory/('step_{:05d}.pt'.format(target)) if target in (2000,5000) else directory/'last.pt'
                    validation = directory/('val_{:05d}'.format(target))
                    if not (validation/'summary.json').exists():
                        estimate = 5400 if solver == 'ddpm' and target in (2000,5000) else (1800 if solver == 'ddpm' else 900)
                        command = [PYTHON, '-u', 'scripts/explore_ag/evaluate.py',
                                   '--checkpoint', str(check), '--output', str(validation),
                                   '--solver', solver, '--steps', str(nfe)]
                        if target not in (2000,5000): command += ['--limit', '24']
                        if not self.complete_resumable(branch+'_val_'+str(target), package, command,
                                                       validation/'summary.json', estimate):
                            self.state['branches'][branch] = {'status': 'inconclusive', 'added_steps': current,
                                                               'limitation': 'Validation incomplete'}
                            self.save()
                            continue
                    if not (validation/'summary.json').exists():
                        self.state['branches'][branch] = {'status': 'budget_insufficient',
                                                           'added_steps': current, 'limitation': 'Partial validation saved'}
                        self.save()
                        continue
                    summary = json.loads((validation/'summary.json').read_text())
                    self.state['branches'][branch] = {'status': 'completed_screen' if target == 5000 else 'running',
                                                       'added_steps': target, 'latest_val': summary,
                                                       'checkpoint': str(check)}
                    if target in (2000,5000):
                        previous = load(directory/'selection.json', {'mean_psnr': float('-inf')})
                        if summary['mean_psnr'] > previous['mean_psnr']:
                            temp = directory/'best.pt.tmp'
                            if temp.exists(): temp.unlink()
                            os.link(check, temp)
                            temp.replace(directory/'best.pt')
                            atomic_json(directory/'selection.json', {'mean_psnr': summary['mean_psnr'],
                                          'step': target, 'checkpoint_sha256': sha256(check),
                                          'selection_split': 'UIEB grouped_v1 validation'})
                    self.save()
                self.write_direction(package, branches)
            if package == 'P1' and self.state['branches'].get('BASE_CONT', {}).get('status') != 'completed_screen':
                for _, dependent_branches in GROUPS[1:]:
                    for branch in dependent_branches:
                        self.state['branches'].setdefault(branch, {'status': 'blocked_dependency',
                            'reason': 'BASE_CONT did not reach a matched 5000-step validation'})
                self.state['status'] = 'waiting_for_baseline'
                self.save()
                return False
        return True

    def freeze_and_benchmark(self, solver, nfe, parent):
        from mpa_diff.data.manifest import read_manifest
        source = read_manifest(str(ROOT/'manifests/uieb_recon_grouped_v1.jsonl'))
        target = read_manifest(str(ROOT/'manifests/lsui_recon_grouped_v1.jsonl'))
        train_hashes = {r['image_sha256'] for r in source if r['split'] in ('train','val')}
        target_test_hashes = {r['image_sha256'] for r in target if r['split'] == 'test'}
        overlap = sorted(train_hashes & target_test_hashes)
        atomic_json(RUN/'cross_dataset_overlap_audit.json',
                    {'source_train_val_images': len(train_hashes), 'lsui_test_images': len(target_test_hashes),
                     'exact_byte_overlap_sha256': overlap,
                     'limitation': 'Exact-byte check does not rule out near-duplicate scenes'})
        if overlap:
            self.state['benchmark_status'] = 'blocked_data_leakage'
            self.save()
            return
        parent_score = self.state['parent']['mean_psnr']
        base = self.state['branches'].get('BASE_CONT', {})
        options = [(str(parent), 'PARENT', parent_score)]
        base_checkpoint = RUN/'branches/BASE_CONT/best.pt'
        if base.get('status') == 'completed_screen' and base_checkpoint.exists():
            base_score = json.loads((RUN/'branches/BASE_CONT/selection.json').read_text())['mean_psnr']
            options.append((str(base_checkpoint), 'BASE_CONT', base_score))
        controls = {'B_PROXY':'BASE_CONT', 'A_SPLIT':'A_COUPLED',
                    'C_ROUTED':'C_FIXED', 'D_PHASE':'D_SOBEL_CONTROL',
                    'F_ROUTED':'F_UNIFORM'}
        for candidate, control in controls.items():
            c = self.state['branches'].get(candidate, {})
            ref = self.state['branches'].get(control, {})
            if c.get('status') != 'completed_screen' or ref.get('status') != 'completed_screen':
                continue
            cs, rs = c['latest_val'], ref['latest_val']
            if cs['mean_psnr'] - rs['mean_psnr'] >= .10 and cs['mean_ssim'] >= rs['mean_ssim']-.002:
                checkpoint = RUN/'branches'/candidate/'best.pt'
                if checkpoint.exists():
                    score = json.loads((checkpoint.parent/'selection.json').read_text())['mean_psnr']
                    options.append((str(checkpoint), candidate, score))
        selected_path, selected_branch, selected_score = max(options, key=lambda item: item[2])
        representatives = [(str(parent), 'PARENT')]
        for package, branches in GROUPS:
            for branch in branches:
                checkpoint = RUN/'branches'/branch/'best.pt'
                if checkpoint.exists() and self.state['branches'].get(branch, {}).get('status') == 'completed_screen':
                    representatives.append((str(checkpoint), branch))
        allowed = list(dict.fromkeys(path for path, _ in representatives))
        freeze = {'frozen_before_test': True, 'frozen_utc': timestamp(),
                  'selection_split': 'UIEB grouped_v1 validation',
                  'selection_rule': 'Quality candidate gate vs matched control; highest source validation PSNR; fallback parent/BASE_CONT',
                  'selected_branch': selected_branch, 'selected_checkpoint': selected_path,
                  'selected_validation_psnr': selected_score, 'solver': solver, 'nfe': nfe,
                  'allowed_checkpoint_sha256': [sha256(p) for p in allowed],
                  'representatives': [{'checkpoint': path, 'branch': branch,
                                       'sha256': sha256(path)} for path, branch in representatives],
                  'limitations': ['LPIPS and predefined robustness gate not yet evaluated; selection is quality-provisional']}
        freeze_path = RUN/'selection_freeze_before_test.json'
        if freeze_path.exists():
            if json.loads(freeze_path.read_text())['selected_checkpoint'] != selected_path:
                raise ValueError('Existing test selection freeze differs; refusing to reselect')
        else:
            atomic_json(freeze_path, freeze)
        self.state['test_selection'] = freeze
        self.save()
        ordered = [(selected_path, selected_branch)] + [item for item in representatives if item[0] != selected_path]
        tasks = [(checkpoint, branch, dataset)
                 for dataset in ('uieb_test', 'lsui_test', 'u45')
                 for checkpoint, branch in ordered]
        coverage = {}
        for checkpoint, branch, dataset in tasks:
            directory = RUN/'benchmark'/branch/dataset
            if (directory/'summary.json').exists():
                coverage[branch+'/'+dataset] = 'completed'
                continue
            estimate = (21000 if dataset == 'lsui_test' else 5400) if solver == 'ddpm' else 1800
            command = [PYTHON, '-u', 'scripts/explore_ag/benchmark.py',
                       '--freeze', str(freeze_path), '--checkpoint', checkpoint,
                       '--dataset', dataset, '--output', str(directory)]
            if not self.complete_resumable('FINAL_'+branch+'_'+dataset, 'FINAL', command,
                                           directory/'summary.json', estimate):
                self.state['benchmark_status'] = 'partial_or_budget_stopped'
                coverage[branch+'/'+dataset] = 'not_run_budget_or_failed'
                self.state['benchmark_coverage'] = coverage
                self.save()
                return
            coverage[branch+'/'+dataset] = 'completed'
            self.state['benchmark_coverage'] = coverage
            self.save()
        self.state['benchmark_status'] = 'completed_for_all_screened_representatives'
        self.save()

    def write_direction(self, package, branches):
        lines = ['# {} minimum screening'.format(package), '',
                 'Protocol: `explore_ag_single_seed_v2_20261003`; common parent and sampler are in `runs/explore_ag_single_seed_v2_20261003/`.', '',
                 '| Branch | Added steps | Validation PSNR | Validation SSIM | Evidence |',
                 '|---|---:|---:|---:|---|']
        for branch in branches:
            state = self.state['branches'].get(branch, {})
            score = state.get('latest_val', {})
            lines.append('| {} | {} | {} | {} | {} |'.format(branch, state.get('added_steps',0),
                         score.get('mean_psnr','missing'), score.get('mean_ssim','missing'),
                         state.get('status','planned')))
        lines += ['', 'Only validation data were used for this screening. Per-image records and checkpoint hashes reside under the branch run directories. Different achieved steps are not a matched ablation.']
        (DOC/('explore_ag_{}.md'.format(package))).write_text('\n'.join(lines)+'\n')

    def finish(self):
        self.state['branches'].setdefault('B_ADAPT', {'status':'blocked_data',
            'reason':'No verified depth labels and original-scene split available'})
        self.state['branches'].setdefault('G_FEATURE', {'status':'blocked_dependency',
            'reason':'No licensed, validated frozen underwater detector backend available'})
        self.state['branches'].setdefault('G0_G1', {'status':'blocked_dependency',
            'reason':'G detector prerequisite unresolved'})
        self.state['status'] = 'screening_finished_with_documented_blocks' if any(
            item.get('status','').startswith('blocked') for item in self.state['branches'].values()
        ) else 'screening_finished'
        self.ledger['status'] = self.state['status']
        self.save()
        self.write_report()

    def write_report(self):
        lines = ['# Single-seed A-G exploration report', '',
                 'Current state: {}. All figures below are source-validation measurements unless explicitly labeled otherwise.'.format(self.state['status']), '',
                 'Common parent: `{}`; evaluation sampler: `{}`.'.format(self.state.get('parent',{}).get('sha256','pending'), self.state.get('eval_protocol','pending')), '',
                 'Device hours charged: {:.3f} / 168; 16 hours reserved for final Benchmark.'.format(self.charged_seconds()/3600), '',
                 '| Direction | Run count | State |', '|---|---:|---|']
        for package, branches in GROUPS:
            finished = sum(self.state['branches'].get(b,{}).get('status') == 'completed_screen' for b in branches)
            lines.append('| {} | {}/{} | {} |'.format(package, finished, len(branches), 'completed_screen' if finished == len(branches) else 'inconclusive'))
        g_finished = sum(self.state['branches'].get(branch, {}).get('status') == 'completed_screen'
                         for branch in ('G_FEATURE', 'G0_G1'))
        lines += ['| E | 1/1 | {} |'.format('completed_screen' if self.state.get('eval_protocol') else 'planned'),
                  '| G | {}/2 | {} |'.format(g_finished,
                      'completed_screen' if g_finished == 2 else self.state['branches'].get('G_FEATURE', {}).get('status', 'planned')), '',
                  'Detailed branch metrics and failures: `runs/explore_ag_single_seed_v2_20261003/branches/`; E0 results: `runs/explore_ag_single_seed_v2_20261003/e0/`.',
                  'Final Benchmark coverage: `{}`. Missing values are not estimated.'.format(
                      self.state.get('benchmark_coverage', {}))]
        (DOC/'EXPLORATION_AG_SINGLE_SEED_V2_REPORT.md').write_text('\n'.join(lines)+'\n')

    def execute(self):
        self.wait_for_legacy()
        solver, nfe, _, selected_result = self.run_e0()
        parent, solver, nfe = self.select_parent(solver, nfe, selected_result)
        profile = RUN/'performance/selection.json'
        if not profile.exists():
            if not self.run('P0_update_profile', 'P0',
                            [PYTHON, '-u', 'scripts/explore_ag/profile.py',
                             '--parent', str(parent), '--output', str(profile.parent)], 1800):
                raise RuntimeError('Real update throughput profile failed')
        micro_batch = load(profile, {})['micro_batch']
        self.state['micro_batch'] = micro_batch
        self.save()
        if not self.train_groups(parent, solver, nfe, micro_batch):
            self.write_report()
            return
        self.freeze_and_benchmark(solver, nfe, parent)
        self.finish()


def torch_step(path):
    import torch
    return torch.load(path, map_location='cpu')['step']


def checkpoint_step(path):
    return torch_step(path) if path.exists() else 0


def main():
    dispatcher = Dispatcher()
    try:
        dispatcher.execute()
    except Exception as exc:
        dispatcher.state['status'] = 'stopped_error'
        dispatcher.state['error'] = type(exc).__name__ + ': ' + str(exc)
        dispatcher.save()
        dispatcher.write_report()
        raise


if __name__ == '__main__':
    main()
