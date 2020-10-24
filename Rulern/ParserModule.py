import os, sys
# this module read rules form a text file, but was cut from the project due to time limitations and genral lack of usefullness
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .RuleModule import Rule

class Parser():
    def __init__(self,filename):
        self.rules = []
        with open(filename) as file:
            self.file_contents = file.read()
            self.parse()

    def print_rules(self):
        for i in self.rules:
            print(i)

    def parse(self):
        i = 0
        for line in self.file_contents.splitlines():
            #print(line.split())
            if line != '' and line.split()[0] != "#":
                if "#" in line.split():
                    temp = line.split('#')[0]
                    #print(temp)
                    stats = line.split('#')[1]
                    temp = temp.strip("If:").split("Then:")
                else:
                    temp = line.strip("If:").split("Then:")
                    #print(temp)
                ant = temp[0].strip()
                #print(ant)
                ant = ant.split("and")
                con = temp[1].strip()
                con = con.split("=")
                in_var_dict = {}
                for a in ant:
                    #print(a)
                    a = a.strip()
                    idx = a.split('[')[1].split(']')[0]
                    var_name = a.split('[')[0]
                    var_val = a.split(' ')[2]
                    op = a.split(' ')[1]
                    #print(op)
                    if var_name in in_var_dict.keys():
                        if type(in_var_dict[var_name][0]) != type([]):
                            in_var_dict[var_name][0] = [in_var_dict[var_name][0]]
                            in_var_dict[var_name][1] = [in_var_dict[var_name][1]]
                            in_var_dict[var_name][2] = [in_var_dict[var_name][2]]
                        in_var_dict[var_name][0].append(-int(idx))
                        in_var_dict[var_name][1].append(op)
                        in_var_dict[var_name][2].append(float(var_val))
                        #print(in_var_dict)
                    else:
                        in_var_dict[var_name] = [[-int(idx)],[op],[float(var_val)]]
                #print(in_var_dict)
                output = con[0].split('[')[0].strip()
                out_var_dict = {}
                out_var_dict["vars"] = {}
                out_var_dict["ops"] = []
                out_var_dict["out_var"] = output
                out_var_dict["bias"] = 0

                for c in con[1].split("+"):
                    if len(c.strip().split(" ")) > 1:
                        #print(c.split('[')[0].strip(" "))
                        temp_var = c.split('[')[0].strip(" ")
                        if temp_var in out_var_dict["vars"].keys():
                            if type(out_var_dict["vars"][temp_var][0]) != type([]):
                                out_var_dict["vars"][temp_var][0] = [out_var_dict["vars"][temp_var][0]]
                                out_var_dict["vars"][temp_var][1] = [out_var_dict["vars"][temp_var][1]]
                            out_var_dict["vars"][temp_var][0].append(-int(c.split('[')[1].split(']')[0]))
                            out_var_dict["vars"][temp_var][1].append(float(c.strip().split(" ")[2]))
                        else:
                            out_var_dict["vars"][temp_var]=[-int(c.split('[')[1].split(']')[0]),float(c.strip().split(" ")[2])]
                    else:
                        #print(c.strip())
                        out_var_dict["bias"] = float(c.strip())

                    #print(c)
                #print(in_var_dict)

                self.rules.append(Rule("R"+str(i),in_var_dict,out_var_dict))
                i+=1
