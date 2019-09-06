import os
path = os.path.split(os.path.realpath(__file__))[0]
pathall = os.path.split(os.path.realpath(__file__))
print(path, pathall)
print("%s   ,,,\n %s" % (path, pathall))
