"""Isolate tests from the real local data: throwaway SQLite DB and PQC key dir (set before app import)."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="hid-tests-")
os.environ["SQLITE_PATH"] = os.path.join(_tmp, "test.db")
os.environ["PQC_KEY_DIR"] = os.path.join(_tmp, "keys")

os.environ["ADMIN_PASSWORD"] = "2468"   # tests use a fixed admin password
