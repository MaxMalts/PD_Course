import os
import sys
import subprocess
import re
import time


def BlockSize(block):
    blockData = subprocess.getoutput("hdfs fsck -blockId " + block)
    addr = re.search(r"Block replica on datanode/rack: ([^/]*)/default", blockData).group(1)
    
    path = subprocess.getoutput("sudo -u hdfsuser ssh hdfsuser@" + addr + " find /dfs/dn/current -name " + block)
    path = path.split("\n")[0]

    size = subprocess.getoutput("sudo -u hdfsuser ssh hdfsuser@" + addr + " wc -c " + path).split(" ")[0]
    return int(size)


fName = "temp.bin"

size = int(sys.argv[1])

os.system("dd if=/dev/zero of=" + fName + " bs=" + str(size) + " count=1" + " > /dev/null 2>&1")
os.system("hdfs dfs -D dfs.replication=1 -put " + fName + " > /dev/null 2>&1")

data = subprocess.getoutput("hdfs fsck /user/hjudge/" + fName + " -files -blocks")
fileBlocks = re.findall(r"blk_[^_]*", data)

fullSize = 0
for block in fileBlocks:
    fullSize += BlockSize(block)

print(fullSize - size)

os.system("hdfs dfs -rm -skipTrash /user/hjudge/" + fName + " > /dev/null 2>&1")