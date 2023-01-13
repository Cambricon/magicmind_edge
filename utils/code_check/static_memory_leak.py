import os
import gitlab
import subprocess
from sh import cppcheck
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
    if new_path.endswith(".cpp"):
        buf = StringIO()
        cppcheck(new_path, _err=buf)
        infos = buf.getvalue().split("\n")
        for info in infos:
            if len(info) < 1:
                break
            info = info.replace("[","").replace("]","")
            file = info.split(":")[0]
            line = info.split(":")[1]
            err = ":".join(info.split(":")[2:])
            mr.discussions.create({'body': err,
                        'position': {
                            'base_sha': diff_ref["base_sha"],
                            'start_sha': diff_ref["start_sha"],
                            'head_sha': diff_ref["head_sha"],
                            'position_type': 'text',
                            'new_line': line,
                            'old_path': change['old_path'],
                            'new_path': change['new_path']}
                        })
