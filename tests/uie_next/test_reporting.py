from pathlib import Path
import zipfile
from uie_next.reporting import archive


def test_review_archive_members_crc_and_sha(tmp_path):
    file=tmp_path/'report.md';file.write_text('actual report\n')
    package=tmp_path/'review.zip';receipt=archive(package,[file],tmp_path)
    assert receipt['crc_verified'] and receipt['all_payload_sha256_verified']
    with zipfile.ZipFile(package) as z:
        assert z.namelist()==['report.md'] and z.read('report.md')==file.read_bytes()
