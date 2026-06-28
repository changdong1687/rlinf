#! /bin/bash
# Run all offline cache-method unit tests (numpy only; no GPU/sim/groot needed).
set -e
TESTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
fail=0
for t in test_cache_common.py test_teacache.py test_bac.py test_bwcache.py test_calibrate.py; do
  echo "==== ${t} ===="
  python "${TESTS_DIR}/${t}" || fail=1
done
exit "${fail}"
