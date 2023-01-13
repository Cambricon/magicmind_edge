import os
import gitlab
import json
import subprocess
from sh import typos
from io import StringIO

gl = gitlab.Gitlab('http://gitlab.software.cambricon.com',private_token=os.getenv("ACCESS_TOKEN"))
project = gl.projects.get(os.getenv("CI_PROJECT_ID"))
mr_req = os.getenv("CI_OPEN_MERGE_REQUESTS")
if not mr_req:
    exit(0)
mr_id = mr_req.split("!")[-1]
mr = project.mergerequests.get(mr_id)

def run_cmd(cmd):
    return subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()

diff_ref = mr.changes()["diff_refs"]

for change in mr.changes()["changes"]:
    new_path = change["new_path"]
    buf = StringIO()
    try:
        typos(new_path, "--format","json", _out=buf)
    except Exception:
        pass
    infos = buf.getvalue()
    if len(infos) == 0:
        continue
    for info in infos.split("\n"):
        if len(info) == 0:
            continue
        res = json.loads(info)
        mr.discussions.create({'body': f"line {res['line_num']}: '{res['typo']}' should be {','.join(res['corrections'])}",
                        'position': {
                            'base_sha': diff_ref["base_sha"],
                            'start_sha': diff_ref["start_sha"],
                            'head_sha': diff_ref["head_sha"],
                            'position_type': 'text',
                            'new_line': res["line_num"],
                            'old_path': change['old_path'],
                            'new_path': change['new_path']}
                        })
