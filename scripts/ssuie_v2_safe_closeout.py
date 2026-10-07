"""Close out V2 without changing its frozen scientific loader or results.

The pinned author's ``net`` directory is a namespace package. The historical
loader cannot inspect its missing __file__ on a second load in one process.
Evict only a verified official namespace entry before delegating to that loader.
Commit/worktree/checkpoint checks and ordinary module collision checks remain
in the original loader. This command never resumes scientific training.
"""
import json
import sys
from pathlib import Path

from uie_next.backbones.ssuie import BackboneUnavailable
from uie_next.records import ROOT, append


def prepare_namespace(source=None):
    source = Path(source or ROOT / 'third_party/ss_uie').resolve()
    module = sys.modules.get('net')
    if module is None or getattr(module, '__file__', None) is not None:
        return False
    spec = getattr(module, '__spec__', None)
    locations = list(getattr(spec, 'submodule_search_locations', None) or [])
    if not locations or any(Path(p).resolve() != source / 'net' for p in locations):
        raise BackboneUnavailable('net namespace collision; refusing closeout')
    # net.model/net.blocks still undergo the frozen loader's file identity check.
    del sys.modules['net']
    return True


def main():
    from uie_next.v2.context import State
    from uie_next.v2.delivery import deliver
    state = State()
    if state.state['scientific_status'] in ['RUNNING', 'INTERRUPTED_RECOVERABLE']:
        raise RuntimeError('Use the authorized dispatcher to resume; closeout denied')
    evicted = prepare_namespace()
    append(state.run / 'commands.jsonl', {
        'argv': sys.argv, 'purpose': 'closeout only; frozen loader preserved',
        'verified_official_namespace_evicted': evicted})
    print(json.dumps(deliver(state), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
