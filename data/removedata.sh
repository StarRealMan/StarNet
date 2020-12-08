#!/bin/bash

echo "Remove all pcd data!"

rm ./savings/*.pcd

echo "Remove all .DS_Store file!"

find . -name '*.DS_Store' -type f -delete