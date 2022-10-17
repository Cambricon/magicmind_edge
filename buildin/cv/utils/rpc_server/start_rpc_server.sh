#!/bin/bash
set -e

export SSHPASS=Hello123

bash_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f ${bash_dir}/bin/edge_server ]; then
    pushd ${bash_dir}
    ./build.sh
    popd
fi

if [ -z "${REMOTE_IP}" ]; then
    echo "REMOTE_IP is not define. example: export REMOTE_IP=192.168.100.131"
    exit 1
fi

if [ -f "/root/.ssh/known_hosts" ]; then
    ssh-keygen -f "/root/.ssh/known_hosts" -R "${REMOTE_IP}"
fi

pid=$(sshpass -e ssh -o "StrictHostKeyChecking=no" root@${REMOTE_IP} ps -ef | grep ./edge_server | awk '{print $1}')
if [ -n "$pid" ]; then
    echo "rpc server already exists. pid is $pid "
else
    sshpass -e scp ${bash_dir}/bin/edge_server root@${REMOTE_IP}:/tmp
    sshpass -e ssh -f root@${REMOTE_IP} "export LD_LIBRARY_PATH=/mps/lib:/usr/local/neuware/edge/lib64:$LD_LIBRARY_PATH; cd /tmp; ./edge_server"
    
    pid=$(sshpass -e ssh -o "StrictHostKeyChecking=no" root@${REMOTE_IP} ps -ef | grep ./edge_server | awk '{print $1}')
    if [ -n "$pid" ]; then
        echo "success to start rpc server. pid is $pid "
    else
        echo "failed to start rpc server"
        exit 1
    fi
fi
