#!/bin/bash
set -e

REMOTE_PRE_CHECK(){
    if [ -z "$REMOTE_IP" ] || [ -z "$REMOTE_DIR" ]; then
        echo "REMOTE_IP or REMOTE_DIR not set, please set it first."
        exit 1 
    fi

    if [ -f "/root/.ssh/known_hosts" ]; then
        ssh-keygen -f "/root/.ssh/known_hosts" -R "${REMOTE_IP}"
    fi
}

REMOTE_MM_RUN(){
    PROJ_DIR=$REMOTE_DIR/${PROJ_ROOT_PATH##*/magicmind_edge}
    echo "@@@ Begin to inference in remote device. remote ip: ${REMOTE_IP}"

    perf_info=$(sshpass -e\
        ssh -o "StrictHostKeyChecking=no" \
        root@${REMOTE_IP} \
        "/bin/bash -c '
        export LD_LIBRARY_PATH=/mps/lib:$LD_LIBRARY_PATH && \
        cd ${PROJ_DIR} && \
        /mps/bin/mm_run $*'")

    fps=`echo -e "$perf_info" | grep "Throughput (qps):"`
    fps=`eval echo "${fps:18}"`
    
    if [ -z "$fps" ] ; then
        echo "failed"
        exit 1
    else
        printf "+--------+--------+\n"
        printf "|%-8s|%-8s|\n" fps ${fps}
        printf "+--------+--------+\n"
    fi
    python ${MAGICMIND_EDGE}/utils/record_result.py --fps ${fps}
}

REMOTE_RUN(){
    REMOTE_PRE_CHECK

    PROJ_DIR=$REMOTE_DIR/${PROJ_ROOT_PATH##*/magicmind_edge}
    EXEC_FILE=$(basename $(readlink -f "$0"))
    LOCAL_WORK_DIR=$(dirname $(readlink -f "$0"))
    WORK_DIR=$REMOTE_DIR/${LOCAL_WORK_DIR##*/magicmind_edge}
    
    echo "@@@ Begin to inference in remote device
    remote ip: ${REMOTE_IP}
    remote root dir: ${REMOTE_DIR}
    remote work dir: ${WORK_DIR}
    remote exec file: ${EXEC_FILE} "
    sshpass -e \
        ssh -o "StrictHostKeyChecking=no" \
        root@${REMOTE_IP} \
        "/bin/bash -c 'cd ${PROJ_DIR} && \
        source env.sh && \
        cd ${WORK_DIR} && \
        bash ${EXEC_FILE} $*'"
}

