import os
output = os.popen('wmic process get description, processid').read()
print(output)