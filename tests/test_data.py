import hashlib
import pytest
from evidence_lens.data import verify_archive, fingerprint


def test_corrupted_cached_archive_fails(tmp_path):
    p = tmp_path / 'data.zip'
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        verify_archive(p)


def test_fingerprint_changes_with_content_revision_and_policy():
    a = fingerprint(['title abstract'], 'revision-a', 256)
    assert a != fingerprint(['changed'], 'revision-a', 256)
    assert a != fingerprint(['title abstract'], 'revision-b', 256)
    assert a != fingerprint(['title abstract'], 'revision-a', 128)
    assert a == fingerprint(['title abstract'], 'revision-a', 256)
