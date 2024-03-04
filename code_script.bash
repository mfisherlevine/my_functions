#! /bin/bash

max_retry=10

for i in $(seq 1 $max_retry)
do
    recent_folder=$(ls ~/.vscode-server/bin/ -t | head -n$i)
    script=$(echo ~/.vscode-server/bin/$recent_folder/bin/remote-cli/code)
    if [[ -z ${script} ]]
    then
        echo "VSCode remote script not found"
        exit 1
    fi
    socket=$(ls /run/user/$UID/vscode-ipc-* -t | head -n$i)
    if [[ -z ${socket} ]]
    then
        echo "VSCode IPC socket not found"
        exit 1
    fi
    export VSCODE_IPC_HOOK_CLI=${socket}
    ${script} $@
    if [ "$?" -eq "0" ]; then
        exit 0
    fi
done

echo "Failed to find valid VS Code window"

