"""Archive and freeze the verified resume before the first formal pilot update."""
import datetime
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *


def main():
    assert sha(PARENT) == PARENT_HASH
    budget = read(RUN/'budget.json')
    assert budget['active'] is None
    assert read(RUN/'cost_prediction.json')['dispatch_allowed']
    assert not (RUN/'pilot').exists(), 'Freeze must precede formal pilot'
    suite = ET.parse(RUN/'tests_resume.xml').getroot()
    assert all(int(s.get('failures', '0')) == 0 and int(s.get('errors', '0')) == 0
               for s in suite.iter('testsuite'))
    tested = sum(int(s.get('tests', '0')) for s in suite.iter('testsuite'))
    selector = read(RUN/'selector_profile_resume.json')
    assert selector['resume_7_plus_13_vs_20']['passed']
    freeze = {'source_sha256': code_hash(), 'training_identity': train_identity(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'parent_sha256': PARENT_HASH, 'protocol_sha256': sha(RUN/'protocol_frozen.yaml'),
              'data_roles_sha256': sha(RUN/'data_roles.json'),
              'runtime_freeze_sha256': sha(RUN/'inference_runtime_freeze.json'),
              'test_report_sha256': sha(RUN/'tests_resume.xml'), 'tests_passed': tested,
              'selector_profile_resume_sha256': sha(RUN/'selector_profile_resume.json'),
              'charged_hours_before_pilot': budget['charged_seconds']/3600,
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'parent_optimizer_reused': False, 'scientific_training_started': False,
              'no_holdout_scores_accessed': True}
    target = RUN/'implementation_freeze_resume_before_pilot.json'
    assert not target.exists(), 'Immutable resume freeze already exists'
    archive = RUN/'artifact_manifest_pre_resume.json'
    if (RUN/'artifact_manifest.json').exists() and not archive.exists():
        shutil.copy2(RUN/'artifact_manifest.json', archive)
    write(target, freeze)
    print(freeze)


if __name__ == '__main__':
    main()
