/usr/bin/gitlab-ci-multi-runner \
    register \
    -n \
    --name "edge_zoo_no_chip_28_concurrency" \
    -u http://xxx.com/ \
    --registration-token xxxxxxxxxxxxx \
    --executor "docker" \
    --docker-image ubuntu18.04 \
    --builds-dir /mount_data/work_dir \
    --docker-tlsverify=false \
    --docker-network-mode "host" \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/models_cache:/mount_data/models_cache' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/datasets:/mount_data/datasets' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/datasets-mini:/mount_data/datasets-mini' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/work_dir:/mount_data/work_dir' \
    --docker-pull-policy if-not-present \
    --tag-list edge_zoo_no_chip_28_concurrency \
    --docker-pull-policy if-not-present



/usr/bin/gitlab-ci-multi-runner \
    register \
    -n \
    --name "ide_runner" \
    -u http://gitlab.software.cambricon.com/ \
    --registration-token Gm833sPCPtwxoHjqhyyH \
    --executor "docker" \
    --docker-image ubuntu18.04 \
    --builds-dir /mount_data/work_dir \
    --docker-tlsverify=false \
    --docker-network-mode "host" \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/models_cache:/mount_data/models_cache' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/datasets:/mount_data/datasets' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/datasets-mini:/mount_data/datasets-mini' \
    --docker-volumes '/home/cambricon/ci/modelzoo_edge/work_dir:/mount_data/work_dir' \
    --docker-volumes '/home/cambricon/ci/dev_env:/dev_env' \
    --docker-pull-policy if-not-present \
    --tag-list ide_runner \
    --docker-pull-policy if-not-present


/dev_env/tools/openvscode/bin/openvscode-server --without-connection-token --host 0.0.0.0 --port {port} --user-data-dir /dev_env/tools/userdata --extensions-dir /dev_env/tools/exts