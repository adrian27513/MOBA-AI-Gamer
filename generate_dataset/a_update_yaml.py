import yaml

f = open("C:/Users/Adrian/PycharmProjects/LeagueHumanPlayerAI/generate_dataset/output/LeagueAI.labels", "r")
num = 0
names = []
for x in f:
    num = num+1
    names.append(x.replace('\n',""))
print ("Number of Objects: " + str(num))
print (names)
f.close()

yamlInfo = {'train': '../data/images/train', 'val': '../data/images/valid', 'nc': num, 'names': names}

with open("data.yaml", "w") as f:
    yaml.dump(yamlInfo, f)
f.close()