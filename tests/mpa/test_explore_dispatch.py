import json

import torch

from scripts.explore_ag import dispatch


def test_parent_is_frozen_and_previous_checkpoint_is_considered(tmp_path, monkeypatch):
    run = tmp_path / 'new'
    legacy = tmp_path / 'old'
    run.mkdir()
    legacy.mkdir()
    monkeypatch.setattr(dispatch, 'RUN', run)
    monkeypatch.setattr(dispatch, 'LEGACY', legacy)
    monkeypatch.setattr(dispatch, 'PREVIOUS', run / 'legacy_step_220000.pt')
    for path, step in ((legacy / 'last.pt', 225000), (legacy / 'best.pt', 85000),
                       (run / 'legacy_step_220000.pt', 220000)):
        torch.save({'step': step}, path)
    latest = run / 'e0/full_latest'
    latest.mkdir(parents=True)
    (latest / 'summary.json').write_text(json.dumps({'ddpm:1000': {'mean_psnr': 24.5,
                                                                  'mean_ssim': .92}}))
    scores = {85000: 24.6, 220000: 24.8}

    dispatcher = dispatch.Dispatcher.__new__(dispatch.Dispatcher)
    dispatcher.state = {'eval_protocol': 'ddpm:1000'}
    dispatcher.save = lambda: None

    def evaluate_candidate(task_id, package, command, summary_path, estimate):
        step = dispatch.torch_step(command[command.index('--checkpoint') + 1])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({'mean_psnr': scores[step], 'mean_ssim': .93}))
        return True

    dispatcher.complete_resumable = evaluate_candidate
    frozen, solver, nfe = dispatcher.select_parent('ddpm', 1000, None)
    assert (solver, nfe) == ('ddpm', 1000)
    assert frozen == run / 'parent_0.pt'
    assert dispatch.torch_step(frozen) == 220000
    receipt = json.loads((run / 'parent_selection.json').read_text())
    assert len(receipt['candidates']) == 3
    assert receipt['previous_complete_checkpoint'] == str(run / 'legacy_step_220000.pt')
    pinned_hash = dispatch.sha256(frozen)
    replacement = legacy / 'replacement.pt'
    torch.save({'step': 230000}, replacement)
    replacement.replace(legacy / 'last.pt')
    assert dispatch.sha256(frozen) == pinned_hash
