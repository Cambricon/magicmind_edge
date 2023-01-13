import requests
import json
import os
import time

job_concurrent_id = 10
job_id = os.getenv("CI_JOB_ID")
headers = {"Private-Token": os.getenv("ACCESS_TOKEN")}
api_url = "http://gitlab.software.cambricon.com/api/v4"
root_dir = os.getenv("CI_PROJECT_DIR")

device_available = False

while not device_available:
    sleep_time = time.perf_counter() % (job_concurrent_id * 15) 
    print(f"not enough device, wating {sleep_time} second ...")
    time.sleep(sleep_time)
    free_edge_ip = os.getenv("CONST_EDGE_IP_LIST").split(",")
    runtime_response = requests.get(f"{api_url}/projects/4261/variables", headers=headers)
    runtime_info = runtime_response.json()
    for env in runtime_response.json():
        if env["key"].startswith("RUNTIME_JOB"):
            run_job_id = env["key"].replace("RUNTIME_JOB_", "")
            job_response = requests.get(f"{api_url}/projects/4261/jobs/{run_job_id}", headers=headers)
            if job_response.json()["status"] == "running":
                print(f"job {run_job_id} use edge device {env['value']}")
                if env["value"] in free_edge_ip:
                    free_edge_ip.remove(env["value"])
            else:
                response = requests.delete(f"{api_url}/projects/4261/variables/{env['key']}",headers=headers)
                if response.status_code == 204:
                    print(f"release edge deive env['value']")
                else:
                    print(f"failed to release edge device {response.content}")
                    if env["value"] in free_edge_ip:
                        free_edge_ip.remove(env["value"])

    if len(free_edge_ip) > 0:
        response = requests.post(f"{api_url}/projects/4261/variables",headers=headers,data={"key":f"RUNTIME_JOB_{job_id}" ,"value": free_edge_ip[0]})
        if response.status_code == 201:
            print(f"success to set variable, {response.content}")
            with open(os.path.join(root_dir, "ci_env.sh") , "w") as f:
                f.write(f"export REMOTE_IP={free_edge_ip[0]}")
            device_available = True
        else:
            print(f"failed to put runtime variable {response.content}")
 
