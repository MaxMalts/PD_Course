import sys
import random

n_lines = int(sys.argv[1])

cur_num = random.randint(1, 5)
for line in sys.stdin:
    if (cur_num == 0):
        cur_num = random.randint(1, 5)
        print()
        
        n_lines -= 1
        if (n_lines == 0):
            break
    
    print(line.strip(), end='')
    
    cur_num -= 1
    if cur_num != 0:
        print(',', end='')