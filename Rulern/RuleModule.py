import numpy as np
import pandas as pd

class Rule:
    def __init__(self,name,ant,con,seq_len = 0):
        self.gen = 0
        self.name = name
        self.ant = ant
        self.con = con
        self.seq_len = seq_len
        self.fitness = 0
        self.correctness = 0
        self.specificity = 0
        self.matched = 0
        self.evaluated = 0
        self.correct = 0
        #self.out_type = "continuous" # continuous or categorical output
        self.kernel = []
        self.df_kernel = None
        #print(con)

    def __str__(self):
        #temp_str =(str(self.name)+" ")
        temp_str = "If: "
        i = 0
        for key in self.ant:
            if type(self.ant[key][0]) != type([]):
                if i > 0:
                    temp_str+= " and "
                temp_str+= " "+ key+"["+str(-self.ant[key][0])+"]"
                for c in self.ant[key][1:]:
                    temp_str+= " "+ str(c)
                i+=1
            else:
                for ins in range(len(self.ant[key][0])):
                    if i > 0:
                        temp_str+= " and "
                    temp_str+= " "+ key+"["+str(-self.ant[key][0][ins])+"]"

                    for c in [self.ant[key][1][ins],self.ant[key][2][ins]]:
                        if type(c) == type([]):
                            temp_str+= " "+ str(c[0])+"["+str(-c[1])+"]"
                        else:
                            temp_str+= " "+ str(c)
                    i+=1
        temp_str+= " Then: "
        i = 0
        cnt = 0
        if "vars" in self.con.keys():
            for var in self.con["vars"]:
                if type(self.con["vars"][var][0]) == type(1):
                    cnt+= 1
                else:
                    for z in range(len(self.con["vars"][var][0])):
                        cnt+= 1
                #print(cnt)
        temp_str+= self.con["out_var"]+"[1] = "

        if "vars" in self.con.keys():
            for var in self.con["vars"]:
                if type(self.con["vars"][var][0]) == type(1):
                    temp_str += str(var) + "["+str(- self.con["vars"][var][0])+"]"+" * "+ str(self.con["vars"][var][1])
                    if i < len(self.con["vars"].keys())-1:
                        temp_str += " + "
                    i += 1
                else:

                    for z in range(len(self.con["vars"][var][0])):
                        #print(self.con["vars"][var][0],z)
                        temp_str += str(var) + "["+str(- self.con["vars"][var][0][z])+"]"+" * "+ str(self.con["vars"][var][1][z])
                        #if i < len(self.con["vars"].keys())-1:
                        if i < cnt - 1:
                            temp_str += " + "
                        i += 1

        temp_str += " + "
        temp_str += str((self.con["bias"]))
        # for key in self.con:
        return temp_str

    def internl_do_op(self,inputs,next_input,ant):
        if ant < 0:
            input_a = next_input.at[-ant-1,key]
        else:
            input_a = inputs.at[idx - ant,key]
        if not self.do_op(self.ant[key][1],self.ant[key][2:],input_a):
            print(key +"["+ str(-ant)+"] " + "Does not meet all antecedents")
            return False
        return True

    def evaluate(self,inputs,next_input ={},con = True,name_var = False,kernel = False,kern_in = None):
        #print(inputs)
        idx = self.seq_len -1
        #print(idx)
        #print(self.ant)
        out = 0
        for key in self.ant: #evaluate antecedents

            if type(self.ant[key][0]) == type(1):

                if self.ant[key][0] < 0:
                    input_a = next_input.at[-self.ant[key][0],key]
                if len(inputs.index) == 1:
                    #print(self.ant[key])
                    input_a = inputs[key]
                    if type(self.ant[key][2]) == type([]):
                        input_b = inputs[self.ant[key][2][0]]
                    else:
                        input_b = self.ant[key][2][0]
                else:
                    input_a = inputs.at[idx - self.ant[key][0],key]

                    if type(self.ant[key][2]) == type([]):
                        input_b = inputs.at[idx - self.ant[key][2][1],self.ant[key][2][0]]
                    else:
                        input_b = self.ant[key][2]

                if not self.do_op(self.ant[key][1],input_b,input_a):
                    return "err0"

            else:
                for z in range(len(self.ant[key][0])):

                    if self.ant[key][0][z] < 0:
                        input_a = next_input.at[-self.ant[key][0][z],key]
                    if len(inputs.index) == 1:
                        #print(self.ant[key],inputs)
                        input_a = inputs[key]
                        if type(self.ant[key][2][z]) == type([]):
                            input_b = inputs[self.ant[key][2][z][0]]
                        else:
                            input_b = self.ant[key][2][z]
                    else:
                        #print(z,self.ant[key])
                        input_a = inputs.at[idx - self.ant[key][0][z],key]

                        if type(self.ant[key][2][z]) == type([]):
                            #print(inputs)
                            input_b = inputs.at[idx - self.ant[key][2][z][1],self.ant[key][2][z][0]]
                        else:
                            #print(self.ant[key][2][z])
                            input_b = self.ant[key][2][z]
                    #print(inputs)
                    #print(self.ant[key][1][z],input_b,input_a)
                    if not self.do_op(self.ant[key][1][z],input_b,input_a):
                        return "err0"

        if not con:
            return True
        else:
            if not kernel:
                #print(self.con["vars"])
                if "vars" in self.con.keys():
                    for key in self.con["vars"].keys():
                        if type(self.con["vars"][key][0]) == type(1):
                            #print(inputs.at[idx - self.con["vars"][key][0],key] * self.con["vars"][key][1])
                            if self.con["vars"][key][0] < 0:
                                out += next_input.at[-self.con["vars"][key][0]-1,key] * self.con["vars"][key][1]
                            else:
                                out += inputs.at[idx - self.con["vars"][key][0],key] * self.con["vars"][key][1]

                        if type(self.con["vars"][key][0]) == type([]):
                            for z in range(len(self.con["vars"][key][0])):

                                if self.con["vars"][key][0][z] < 0:
                                    out += next_input.at[-self.con["vars"][key][0][z],key] * self.con["vars"][key][1][z]
                                if len(inputs) == 1:
                                    out += inputs[key] * self.con["vars"][key][1][z]
                                else:
                                    out += inputs.at[idx - self.con["vars"][key][0][z],key] * self.con["vars"][key][1][z]
                                #print(inputs.at[idx - self.con["vars"][key][0][z],key] * self.con["vars"][key][1][z])
                                #out += inputs.at[idx - self.con["vars"][key][0][z],key] * self.con["vars"][key][1][z]
                    out += self.con["bias"]
                else:
                    out += self.con["bias"]
            else:
                out = self.evaluate_with_kernel(kern_in)
                #print(out)

            if name_var:
                return [out,self.con["out_var"]]
            else:
                return out

    def make_kernel(self,template_input):
        #print(template_input*0)
        self.df_kernel = template_input*0

        #print(template_input)

        for key in self.con["vars"]:
            for index in range(len(self.con["vars"][key][0])):
                #print(self.con["vars"][key][1][index])
                self.df_kernel[key][self.con["vars"][key][0][index]]= self.con["vars"][key][1][index]

        #print(self.df_kernel)
        self.kernel = self.df_kernel.values.ravel()
        #print(self.kernel)

    def inverse_kernel(self,new_kernel):
        adj = self.seq_len -1
        shape = self.df_kernel.shape

        new_vars = pd.DataFrame(np.array(new_kernel).reshape(shape),columns= self.df_kernel.columns)
        #print(new_vars)
        self.kernel = new_kernel

        new_vars_dict = new_vars.to_dict()
        #print(new_vars_dict)

        #print(self.con)
        #self.con = {}
        for key in new_vars_dict:
            self.con["vars"][key] = [[],[]]
            for index in new_vars_dict[key]:
                if new_vars_dict[key][index] != 0:
                    self.con["vars"][key][0].append(index)
                    self.con["vars"][key][1].append(new_vars_dict[key][index])




    def evaluate_with_kernel(self,inputs):
        #print(np.transpose(inputs),self.kernel)
        # print(np.matmul((inputs),self.kernel) + self.con["bias"])
        return np.dot(inputs,self.kernel) + self.con["bias"]



    def do_op(self,op,varii,vari):

        var = float(vari)
        var1 = float(varii)
        #print(var,op,var1)
        if op == '>':
            return (var > var1)
        if op == '>=':
            return (var >= var1)
        if op == '<':
            return (var < var1)
        if op == '<=':
            return (var <= var1)
        if op == '==':
            return (var == var1)
        if op == '!=':
            return (var != var1)
        else:
            return False
