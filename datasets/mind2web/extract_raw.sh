#!/bin/bash

wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz

extracted_dir=$(ls -d globusconnectpersonal-*/)
if [ -z "$extracted_dir" ]; then
    echo "Could not find the extracted globusconnectpersonal directory"
fi
echo "$extracted_dir"

cd "$extracted_dir" || exit
./globusconnectpersonal -setup

globus_endpoint_local_id=$(globus endpoint local-id)
echo "$globus_endpoint_local_id"

./globusconnectpersonal -start &

cd ..
python extract_raw.py "$globus_endpoint_local_id"
