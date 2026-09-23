import json
import subprocess
import time
import sys

remote_check = """
import glob
from collections import Counter
import sys

run_name = sys.argv[1]
files = sorted(glob.glob("/home/gzw/projects/MPUS_GA/results_" + run_name + "/*/*.offline.json"))
print("TOTAL:", len(files))
c = Counter(f.split('/')[-2] for f in files)
for k in sorted(c.keys()):
    print("  Direction " + k + ": " + str(c[k]) + " folds")
"""

for host, run in [('xju', 'cbst_quota_xju_fixed_20260922_s43_v1'), ('csu', 'cbst_quota_csu_best_20260922_s43_v1')]:
    print(f"=== {host} ({run}) ===")
    for attempt in range(3):
        res = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', host, 'python3', '-c', remote_check, run], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout)
            break
        time.sleep(2)
