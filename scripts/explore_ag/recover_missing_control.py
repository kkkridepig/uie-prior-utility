"""One bounded A_COUPLED retry, serialized behind the existing dispatcher.

Never changes the frozen test selection; uses the original cumulative ledger.
"""
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explore_ag import dispatch


def main():
    receipt = dispatch.RUN/'control_recovery.json'
    # This separate lock prevents duplicate waiting recovery processes.
    with (dispatch.RUN/'control_recovery.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = dispatch.load(receipt, {})
        if prior.get('status') in ('completed', 'failed', 'not_needed'):
            return
        while True:
            try:
                runner = dispatch.Dispatcher()
                break
            except BlockingIOError:
                time.sleep(30)
        state = runner.state
        target = state['branches'].get('A_COUPLED', {})
        if target.get('status') == 'completed_screen':
            dispatch.atomic_json(receipt, {'status': 'not_needed', 'utc': dispatch.timestamp()})
            return
        failures = [e for e in runner.ledger['events']
                    if e.get('event') == 'task_end' and e.get('id') == 'A_COUPLED_train_1000']
        if not prior and (len(failures) != 1 or failures[0].get('exit_code') != -11):
            raise RuntimeError('Retry authorization restricted to the observed single startup SIGSEGV')
        if not prior and (dispatch.RUN/'branches/A_COUPLED/last.pt').exists():
            raise RuntimeError('Unexpected checkpoint: inspect before retrying')
        parent = dispatch.RUN/'parent_0.pt'
        if not parent.exists() or not state.get('parent'):
            raise RuntimeError('Frozen common parent is missing')
        original_status = prior.get('original_status', state['status'])
        dispatch.atomic_json(receipt, {'status': 'running', 'utc': dispatch.timestamp(),
            'original_status': original_status, 'original_failure': failures[0],
            'test_selection_policy': 'Never reselect using supplemental results; source validation only'})
        # Reuse the original train/evaluate code and its resume hashes unchanged.
        original_groups = dispatch.GROUPS
        dispatch.GROUPS = [('A', ['A_COUPLED', 'A_SPLIT'])]
        state['branches']['A_COUPLED'] = {'status': 'recovery_pending',
            'added_steps': dispatch.checkpoint_step(dispatch.RUN/'branches/A_COUPLED/last.pt')}
        solver, nfe = state['eval_protocol'].split(':')
        runner.save()
        runner.train_groups(parent, solver, int(nfe), state['micro_batch'])
        dispatch.GROUPS = original_groups
        result = state['branches']['A_COUPLED']
        dispatch.atomic_json(receipt, {'status': 'completed' if result['status'] == 'completed_screen' else 'failed',
            'utc': dispatch.timestamp(), 'original_status': original_status,
            'result': result, 'original_failure': failures[0],
            'test_selection_policy': 'Existing selection freeze and test results unchanged'})
        state['status'] = original_status
        runner.save()
        from scripts.explore_ag.report_watch import render_progress
        render_progress()


if __name__ == '__main__':
    main()
