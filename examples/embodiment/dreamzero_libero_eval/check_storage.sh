#! /bin/bash
# Inspect server storage: capacity, free space, SSD/NVMe vs HDD, and candidate fast
# local dirs for staging checkpoints. Optionally pass paths to also report their size
# and which disk they live on.
#
# Usage:
#   bash check_storage.sh
#   bash check_storage.sh /path/to/ckpt_dir /path/to/umt5-xxl   # also du + findmnt these

set -u

hr() { printf '\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "1. Filesystems: capacity / free / type (df -hT)"
df -hT 2>/dev/null || df -h

hr "2. Disks: SSD/NVMe vs HDD  (ROTA=0 -> SSD/NVMe, ROTA=1 -> HDD)"
if have lsblk; then
    lsblk -d -o NAME,ROTA,SIZE,TYPE,MODEL,MOUNTPOINT 2>/dev/null \
        || lsblk -d -o NAME,ROTA,SIZE,TYPE
else
    echo "lsblk not available; checking /sys/block rotational flags:"
    for d in /sys/block/*/queue/rotational; do
        dev=$(echo "$d" | cut -d/ -f4)
        rota=$(cat "$d" 2>/dev/null)
        echo "  $dev rotational=$rota ($([ "$rota" = 0 ] && echo SSD/NVMe || echo HDD))"
    done
fi

if have nvme; then
    hr "2b. NVMe devices (nvme list)"
    nvme list 2>/dev/null || echo "  (nvme list failed / needs sudo)"
fi

hr "3. Candidate fast local dirs (free space)"
CANDIDATES=(/dev/shm /tmp "$HOME" /local /scratch /raid /data)
for d in "${CANDIDATES[@]}"; do
    [ -d "$d" ] || continue
    line=$(df -hT "$d" 2>/dev/null | tail -1)
    echo "  $d -> $line"
done
echo
echo "  Note: /dev/shm is tmpfs (RAM-backed, fastest, but volatile and uses memory)."

hr "4. Paths you passed (size + which disk)"
if [ "$#" -eq 0 ]; then
    echo "  (none — pass paths as args to also see 'du -sh' + the disk they live on)"
else
    for p in "$@"; do
        if [ -e "$p" ]; then
            echo "--- $p"
            du -sh "$p" 2>/dev/null
            if have findmnt; then
                findmnt -T "$p" -o SOURCE,FSTYPE,SIZE,AVAIL,TARGET 2>/dev/null
            else
                df -hT "$p" 2>/dev/null | tail -1
            fi
            # SSD/HDD of that path's backing device
            src=$(df -P "$p" 2>/dev/null | tail -1 | awk '{print $1}')
            base=$(basename "$src" | sed 's/[0-9]*$//; s/p$//')
            rota_file="/sys/block/${base}/queue/rotational"
            [ -f "$rota_file" ] && echo "  backing dev $base rotational=$(cat "$rota_file") ($([ "$(cat "$rota_file")" = 0 ] && echo SSD/NVMe || echo HDD))"
        else
            echo "--- $p  (does not exist)"
        fi
    done
fi

hr "Done"
echo "Tip: stage ckpts on a dir that is (a) ROTA=0 / SSD-NVMe and (b) has enough Avail,"
echo "     then point MODEL_PATH / TOKENIZER_PATH there. /dev/shm is fastest if RAM allows."
