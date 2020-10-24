import numpy as np
import os, sys
import random
import copy
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .RuleModule import Rule
from sklearn.linear_model import LinearRegression

def choose_random(inp,mu = None, sigma = None, neq = ["!="]):                   # choose at random form list
    rand = random.choice(inp)
    while rand == None or (rand in neq):
        rand = random.choice(inp)
    return rand

class GenOps():

    def __init__(self,inp_s,out_s,in_shape,out_shape,max_pop):

        self.Rule_module = Rule("xx",dict(),dict())

        self.new_ant_chance = 0.01                                              # chance of new antecedent and new consequent
        self.new_con_chance = 0.01

        self.mu = 0.0                                                           # mu and sigma of the normal distibution governing the genetic algorithms
        self.sigma = 0.5
        self.interval = 0.05                                                    # the interval of the generated ranges

        self.input_shape = in_shape                                             # input shape
        self.output_shape = out_shape                                           # output shape
        self.input_template = None                                              # input template, used for kernel generation and rule generation

        self.input_names = inp_s                                                # input names list of strings
        self.output_names = out_s                                               # output names list of strings

        self.max_ranges = 2                                                     # max number of ranges a rule can have = > i1 > 0.5 and i1 < 1.0
        self.max_attribute_comp = 2                                             # max number of attribute comparisons a rule can have = > i0 >= i1
        self.max_comp = 0                                                       # max number of attribute comparisons to a cosntant a rule can have = > i0 >= 0.5
        self.max_output_attributes = 3                                          # max number of ouput atributes excl. bias => i1*0.5 + i2*0.5

        self.mag = 1
        self.mag_output = 1


        self.max_pop = max_pop                                                  # max population

        self.seq_len = in_shape[1]                                              # len of sequence, for sequential data
        self.seq_len_con = out_shape[1]

        self.op_list = [">=","<=","!=","=="]#,"<"">",                           # list of rule connectives you can choose from


    def BasicGA(self,rule_set,num_new_rule = 5,tournament_size = 0.5):          # genetic algorithms

        tournament_size = int(tournament_size*len(rule_set))
        original_len = len(rule_set)
        indexes = list(range(tournament_size))
        parents_names = []

        child_rules = []
        if len(rule_set) == 1:
            for n in range(num_new_rule*2):
                parents_names.append(rule_set[0].name)

        else:
            for n in range(num_new_rule*2):

                fitness_arr = []

                for i in range(len(indexes)):

                    indexes[i] = choose_random(list(range(original_len)),neq = indexes)

                for rule in rule_set:
                    fitness_arr.append(np.array([rule.name,rule.fitness]))

                new_arr = []

                for i in indexes:
                    new_arr.append(fitness_arr[i])

                fitness_arr = np.array(new_arr)
                #print(fitness_arr)
                fitness_arr =fitness_arr[np.argsort(fitness_arr.reshape(len(fitness_arr),2)[:, 1]),:]
                parents_names.append(fitness_arr[-1][0])

        # print(parents)
        i = 0
        for couple in np.array(parents_names).reshape(num_new_rule,2):
            parents = []
            #print(couple)

            for rule in rule_set:
                if rule.name == str(couple[0]) or rule.name == str(couple[1]):
                    rule_copy = copy.deepcopy(rule)
                    parents.append(rule_copy)

                    if str(couple[0]) == str(couple[1]):
                        parents.append(rule_copy)
                        ##print([rule_copy,rule])

            #print(parents)
            child_rules.append(self.mutate(self.make_child(parents[0],parents[1],i)))

            i+=1

        new_rb = child_rules + rule_set
        self.check_names(new_rb)
        return new_rb


    def check_names(self,rb):                                                   #check if names in a rule base repeat, if they do change them

        for rule in rb:
            cnt = 0
            for rule2 in rb:
                if rule2.name == rule.name:
                    if cnt > 0:
                        rule2.name = rule2.name + "("+str(cnt)+")"
                        if len(rule2.name) > 30:
                            rule2.name = "RR("+str(cnt)+"|"+str(time.time())+")"
                    cnt+=1
        #print(rb)


    def random_ant(self,neq_list  = ["!=","==","<","<="]):                      # generate a random antecedent

        rand_key = choose_random(self.input_names)
        if self.seq_len == 1:
            random_index = 0
        else:
            random_index = choose_random(range(self.seq_len))
        random_op = choose_random(self.op_list,neq = neq_list)
        random_comp = random.randint(-100,100)/100

        return rand_key,random_index,random_op,random_comp


    def add_ant(self,ant,key,index,op,comp):                                    # add antecedent to the dictionary

        temp_arr = [[index],[op],[comp]]
        #print(ant)
        if key in ant.keys():
            ant[key][0].append(index)
            ant[key][1].append(op)
            ant[key][2].append(comp)
        else:
            ant[key] = temp_arr

    def random_con(self):                                                       # generate a radom conseqent

        rand_key = choose_random(self.input_names)
        random_index = choose_random(range(self.seq_len))
        random_mod = random.randint(-100,100)/100

        return rand_key,random_index,random_mod


    def add_con(self,con_vars,key,index,mod):                                   # add consequent to the rules dictionary

        if key in con_vars.keys():
            #print(cnt," activated",rand_key,ant[rand_key])
            if index != con_vars[key][0][0]:
                con_vars[key][0].append(index)
                con_vars[key][1].append(mod)
            # else:
            #     #print(con_vars[key])
            #     con_vars[key][0]
            #     con_vars[key][1][0] += mod
            #     #print(con_vars[key])

            #print(cnt," post",rand_key,ant[rand_key])
        else:
            con_vars[key] = [[index],[mod]]


    def add_ant_range(self,ant,inp = None,single = False):                      # add the antecedent and make a range, if inp != None, then make sure it macthes the input

        #print("hi")
        if type(inp) == type(None):

            rand_key,random_index,random_op,random_comp = self.random_ant()

            if single:
                random_op = choose_random(self.op_list,neq = ["!="])
                self.add_ant(ant,rand_key,random_index,random_op,random_comp)

            else:
                self.add_ant(ant,rand_key,random_index,random_op,random_comp - self.interval*random.randint(5,100)/100)
                next_op = choose_random(self.op_list,neq = ["!=","==",">",">="])
                self.add_ant(ant,rand_key,random_index,next_op,random_comp + self.interval*random.randint(5,100)/100)

        if type(inp) != type(None):

            rand_key,random_index,random_op,_ = self.random_ant()

            if single:
                random_op = choose_random(self.op_list,neq = ["!="])
                # print(random_op)
                if random_op in [">",">="]:
                    # print("hi1")
                    self.add_ant(ant,rand_key,random_index,random_op,inp.at[(self.seq_len -1) - random_index, rand_key] - self.interval*random.randint(5,100)/100)
                elif random_op in ["<","<="]:
                    # print("hi2")
                    self.add_ant(ant,rand_key,random_index,random_op,inp.at[(self.seq_len -1) - random_index, rand_key] + self.interval*random.randint(5,100)/100)
                elif random_op == "==":
                    self.add_ant(ant,rand_key,random_index,random_op,inp.at[(self.seq_len -1) - random_index, rand_key])

            else:
                self.add_ant(ant,rand_key,random_index,random_op,inp.at[(self.seq_len -1) - random_index, rand_key] - self.interval*random.randint(5,100)/100)
                next_op = choose_random(self.op_list,neq = ["!=","==",">",">="])
                #print(random_op,next_op)
                self.add_ant(ant,rand_key,random_index,next_op,inp.at[(self.seq_len -1) - random_index, rand_key] + self.interval*random.randint(5,100)/100)
                #print(ant)

        #print(random_index, rand_key)
        #self.add_ant(ant,rand_key,random_index,next_op,random_comp - self.interval*random.randint(5,100)/100)


    def add_ant_comp(self,ant,inp = None):                                      # as above but add a compariosn
        #print(inp)
        if type(inp) == type(None):

            rand_key,random_index,random_op,random_comp = self.random_ant(neq_list = [])
            rand_key1,random_index1,_,_ = self.random_ant()

        if type(inp) != type(None):

            rand_key,random_index,random_op,random_comp = self.random_ant(neq_list = [])
            rand_key1,random_index1,_,_ = self.random_ant()

            #print(random_index,random_index1,(self.seq_len -1) - random_index1)
            while not self.Rule_module.do_op(random_op,inp.at[(self.seq_len -1) - random_index1, rand_key1],
                                                                inp.at[(self.seq_len -1) - random_index, rand_key]):

                rand_key,random_index,random_op,random_comp = self.random_ant(neq_list = [])
                rand_key1,random_index1,_,_ = self.random_ant()
        #print(random_op)
        self.add_ant(ant,rand_key,random_index,random_op,[rand_key1,random_index1])

    def FitSubSet(self,samples,target,rule_base,masking = True):                # fit a subset of rules to data
        #print(samples,target)

        for rule in rule_base:
            self.MLLSQ_FitRule(samples,target,rule,mask = masking)


    def MLLSQ_FitRule(self,samples,target,rule, mask = True):                   # perform fitting using sklearn https://scikit-learn.org/stable/modules/linear_model.html
        #print("Fiting")
        inp_len = self.input_shape[0]*self.input_shape[1]
        # print(rule)
        # print(rule.kernel)
        #print(samples)
        #mask = False
        rule.make_kernel(self.input_template)
        if mask:
            mask_idx = []
            for i in range(len(rule.kernel)):
                if rule.kernel[i] != 0:
                    mask_idx.append(1)
                else:
                    mask_idx.append(0)
            mask_idx = np.diag(np.transpose(mask_idx))
            X = np.array(samples).dot(mask)
            #print(X)


            #X = np.array(X_temp).reshape(len(samples),len(mask_idx))
            #print(X)

        else:
            mask_idx = np.nonzero(np.ones(np.array(rule.kernel).shape))[0]
            X = np.array(samples).reshape(len(samples),inp_len)

        regression = LinearRegression()
        linear_model = regression.fit(X,target)

        if type(linear_model.coef_[0]) != type(np.array([])) and type(linear_model.coef_[0]) != type([]):
            beta = linear_model.coef_
        else:
            beta = linear_model.coef_[0]

        betai = np.zeros(np.array(rule.kernel).shape)
        for i in range(len(mask_idx)):
            #print(mask_idx[i])
            betai[mask_idx[i]] = beta[i]

        rule.con["bias"] = linear_model.intercept_
        #print(betai)
        rule.inverse_kernel(betai)


    def init_pop(self,pop_size = 10, rule_base = [] ,input = None,output = None):   #  initalise random population, if inp != None, create matching rules (covering)

        cnt = 0
        while len(rule_base) < pop_size:
            #print(len(rule_base))
            ant = {}
            con = {}
            while len(ant.keys()) == 0:
                #print("Loop")
                if self.max_ranges != 0:
                    #print("Loop1")
                    randominteger = random.randint(0,self.max_ranges)
                    for i in range(randominteger):
                        self.add_ant_range(ant,inp = input,single = False)

                if self.max_attribute_comp != 0:
                    #print("Loop2")
                    randominteger = random.randint(0,self.max_attribute_comp)
                    for i in range(randominteger):
                        self.add_ant_comp(ant,inp = input)

                if self.max_comp != 0:
                    #print("Loop3")
                    randominteger = random.randint(0,self.max_comp)
                    for i in range(randominteger):
                        self.add_ant_range(ant,inp = input,single = True)
                        #print(ant)

            temp_dict = {}

            if self.max_output_attributes > 0:
                for i in range(random.randint(1,self.max_output_attributes)):

                    rand_key,random_index,random_mod = self.random_con()
                    self.add_con(temp_dict,rand_key,random_index,random_mod)

                con = {
                    "out_var":choose_random(self.output_names),
                    "vars":temp_dict,
                    "bias":random.randint(-100,100)/100}
            else:
                con = {
                    "out_var":choose_random(self.output_names),
                    "bias":random.randint(-100,100)/100}

            rule = Rule("R"+str(cnt),ant,con,self.seq_len)
            #print(rule)

            if type(input) != type(None) and type(output) != type(None):

                pred = rule.evaluate(input,name_var = True)
                #print(pred)
                out = float(pred[0])
                out_key = pred[1]
                #print(out_key)

                if out != 0.0:

                    #print(m,out_key,out)
                    if self.max_output_attributes > 0:
                        m = output[out_key].values[0]/out
                        rule.con["bias"] *= m
                        for key in rule.con["vars"]:
                            for i in range(len(rule.con["vars"][key][0])):
                                rule.con["vars"][key][1][i] *= m
                    else:
                        rule.con["bias"] = output[out_key].values[0]
                else:
                    m = 0.0
                    #print(m,out_key,out)
                    rule.con["bias"] *= m
                    if self.max_output_attributes > 0:
                        for key in rule.con["vars"]:
                            for i in range(len(rule.con["vars"][key][0])):
                                rule.con["vars"][key][1][i] *= m

                if self.max_output_attributes > 0:
                    rule.make_kernel(self.input_template)

            else:
                if self.max_output_attributes > 0:
                    rule.make_kernel(self.input_template)


            rule_base.append(rule)
            cnt+=1
        # for r in rule_base:
        #     print(r)
        #print(len(rule_base))
        #return rule_base


    def make_child(self,parent_r1,parent_r2,i = 0):                             # combine and mutate 2 using crossover

        rchild = Rule("",parent_r1.ant,parent_r2.con,self.seq_len)
        rchild.name = "R("+str(i)+parent_r1.name +"-"+ parent_r1.name+")"
        if len(rchild.name) > 50:
            rchild.name = "RR("+str(i)+"|"+str(time.time())+")"
        #print(rchild)
        return rchild


    def mutate(self,rule):                                                      # mutate the rules parameters

        ant_cnt = 0
        con_cnt = 0

        for key in rule.ant:
            for val in rule.ant[key]:
                ant_cnt+=1

        if self.max_output_attributes > 0:
            for key in rule.con["vars"]:
                for val in rule.con["vars"][key]:
                    con_cnt+=1

        temp_ant = rule.ant.copy()

        for key in temp_ant:

            for i in range(len(rule.ant[key][0])):

                if type(rule.ant[key][2][i]) != type([]):
                    rule.ant[key][2][i] += np.random.normal(self.mu, self.sigma)*self.mag

            if ant_cnt < self.max_ranges:
                if random.randint(0,101)/100 >= self.new_ant_chance:

                    rand_key,random_index,random_op,random_comp = self.random_ant()
                    self.add_ant(rule.ant,rand_key,random_index,random_op,random_comp)

        rule.con["bias"] += np.random.normal(self.mu, self.sigma)*self.mag_output

        if self.max_output_attributes > 0:

            for key in rule.con["vars"]:

                for i in range(len(rule.con["vars"][key][0])):

                    rule.con["vars"][key][1][i] += np.random.normal(self.mu, self.sigma)*self.mag_output

            if con_cnt < self.max_output_attributes:
                if random.randint(0,101)/100 >= self.new_con_chance:
                    rand_key,random_index,random_mod = self.random_con()
                    self.add_con(rule.con["vars"],key,random_index,random_mod)

        if self.max_output_attributes > 0:
            rule.make_kernel(self.input_template)

        return rule
