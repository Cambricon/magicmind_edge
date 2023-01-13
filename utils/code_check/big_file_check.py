import os
import gitlab

gl = gitlab.Gitlab('http://gitlab.software.cambricon.com',private_token=os.getenv("ACCESS_TOKEN"))
project = gl.projects.get(os.getenv("CI_PROJECT_ID"))
mr_req = os.getenv("CI_OPEN_MERGE_REQUESTS")
if not mr_req:
    exit(0)
mr_id = mr_req.split("!")[-1]
mr = project.mergerequests.get(mr_id)

def bytes_trans(bytes):
  if bytes < 1024:
    bytes = str(round(bytes, 2)) + ' B'
  elif bytes >= 1024 and bytes < 1024 * 1024:
    bytes = str(round(bytes / 1024, 2)) + ' KB'
  elif bytes >= 1024 * 1024 and bytes < 1024 * 1024 * 1024:
    bytes = str(round(bytes / 1024 / 1024, 2)) + ' MB'
  elif bytes >= 1024 * 1024 * 1024 and bytes < 1024 * 1024 * 1024 * 1024:
    bytes = str(round(bytes / 1024 / 1024 / 1024, 2)) + ' GB'
  return bytes

msgs = []

for change in mr.changes()["changes"]:
    file_path = change["new_path"]
    if os.path.exists(file_path):
        stats = os.stat(change["new_path"])
        file_size_bytes = stats.st_size
        if file_size_bytes > 100 * 1024: # 100 KB
            msgs.append(f"[{file_path}]: filesize({bytes_trans(file_size_bytes)}) too big, please check it.")

if len(msgs) > 0:
    discs = discussion = mr.discussions.list()
    for dis in discs:
        for note in dis.attributes["notes"]:
            if "too big, please check it." in note["body"]:
                if note["body"] in msgs:
                    msgs.remove(note["body"])
    if len(msgs) > 0:
        discussion = mr.discussions.create({'body': '\n'.join(msgs)})
    
