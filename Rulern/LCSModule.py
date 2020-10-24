import os, sys
import pandas as pd
import numpy as np
import random
import time
from timeit import default_timer as timer
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .RuleModule import Rule
from .ParserModule import Parser
from .GeneticAlgorithmModule import GenOps
from sklearn.metrics import mean_absolute_error,accuracy_score

def choose_random(self,inp,mu = None, sigma = None, neq = []):
    rand = random.choice(inp)
    while rand == None or (rand in neq):
        rand = random.choice(inp)
    return rand

def chck_arr_eq(arr1,arr2):
    if (type(arr1) == type(dict().keys()) or type(arr1) == type(list())) and (type(arr2) == type(dict().keys()) or type(arr2) == type(list())) and (type(arr2) == type(arr1)):
        return collections.Counter(arr1) == collections.Counter(arr2)
    else:
        return False

class LCS():
    def __init__(self,input_s,output_s,rule_base = None,max_pop = 100):


        self.max_pop = max_pop

        if rule_base != None:
            self.pars  = Parser(rule_base)
            self.rules = self.pars.rules.copy()
        else:
            self.pars = None
            self.rules = []

        self.covering_threshold = 10                                            # how many rules must match an instance, if below initiate covering
        self.discarder_rules = []                                               # store discarded rules
        self.input_names = []                                                   # list of input columns
        self.output_names = []                                                  # list of outout columns
        self.input_shape = input_s                                              # input shape
        self.output_shape = output_s                                            # output shape
        self.min_age = 3                                                        # how long before rules qualify for purge
        self.purge_threshold = 0.9                                              # minimum fitness of rules in order to not be purged
        self.fitness_weights =[1,1,1]                                           # weights for c1 c2 and c3 fitness factors
        self.eval_match_ratio = 0.001                                           # ratio of how many times a rule matched versus how many times it was evaluated, prevents overly specific rules
        self.type = "Continuous"                                                # what type of classifcation "Multi-CLass" and "Class"



        self.seq_len = self.input_shape[1]

        self.op_list = [">",">=","<","<=","!=","=="]

        if len(self.input_names) < self.input_shape[0]:
            j = 0
            while len(self.input_names) < self.input_shape[0]:
                temp_str = "inp"+str(j)
                self.input_names.append(temp_str)
                j+=1

        if len(self.output_names) <= self.output_shape[0]:
            j = 0
            while len(self.output_names) < self.output_shape[0]:
                temp_str = "out"+str(j)
                self.output_names.append(temp_str)
                j+=1

        #print(self.input_names,self.output_names)

    def initGA(self):
        self.GA = GenOps(self.input_names,
                        self.output_names,
                        self.input_shape,
                        self.output_shape,
                        self.max_pop)


    def choose_random(self,inp,mu = None, sigma = None, neq = []):
        rand = random.choice(inp)
        while rand == None or (rand in neq):
            rand = random.choice(inp)
        return rand


    def init_rules(self,pop_size = 1000):                                       # initilise random rules
        self.GA.init_pop(pop_size,rule_base  = self.rules)


    def subsume(self,rule_set = [], subsumtion_treshold = 0.05):                # subsume rules that are too similar
        #print("started")
        list_to_be_removed = []

        for rule1 in rule_set:
            for rule2 in rule_set:
                if rule1.name != rule2.name:
                    if chck_arr_eq(rule1.ant.keys(),rule2.ant.keys()):
                        for key in rule1.ant.keys():

                            if chck_arr_eq(rule1.ant[key][0],rule2.ant[key][0]) and chck_arr_eq(rule1.ant[key][1],rule2.ant[key][1]):
                                    all_conf = True

                                    for idx in range(len(rule1.ant[key][2])):
                                        if type(rule1.ant[key][2][idx]) != type([]) and type(rule2.ant[key][2][idx]) != type([]) :

                                            value = rule1.ant[key][2][idx]
                                            value2 = rule2.ant[key][2][idx]
                                            # print(value,value2)
                                            # print(type(value),type(value2))
                                            deviation = subsumtion_treshold * value
                                            if np.abs(float(value)-float(value2)) > deviation:
                                                all_conf = False
                                        else:
                                            all_conf = False
                                            #all_conf = chck_arr_eq(rule1.ant[key][2][idx],rule2.ant[key][2][idx])

                                    if all_conf:

                                        if rule1.fitness >= rule2.fitness:
                                            list_to_be_removed.append(rule2.name)
                                        else:
                                            list_to_be_removed.append(rule1.name)

        rules = []

        for rule in rule_set:
            if rule.name not in list_to_be_removed:
                rules.append(rule)
        return rules


    def covering(self,instance,output_inst,pop  = 10,kernel_in = None):         # generate rules matching an input instance
        cover_rules =[]
        #print(len(cover_rules))
        self.GA.init_pop(pop_size = pop,input = instance,output=output_inst,rule_base = cover_rules)
        # for rule in cover_rules:
        #     print(rule)
        for rule in cover_rules:
            rule.name += "(C|"+str(time.time())+")"
        self.rules += cover_rules
        self.GA.check_names(self.rules)

    # def MLLSQ_batch_covering(self,instances,outputs,buckets = 3):
    #     #print(outputs)
    #
    #     #take data and bucket it

    def evaluate_data(self,instances,kernel_mode = True):                       # evaluate on data set, use this when evaluating!
        pred_arr = []
        active_arr = []
        #print(type(instances))

        for indx in range(len(instances)-(self.seq_len-1)):
            #print("hi")

            if type(instances) == type(pd.DataFrame()):
                if self.seq_len == 1:
                    instance = instances.iloc[[indx]].reset_index(drop = True)
                if self.seq_len > 1:
                    #print(len(instances)-(self.seq_len),indx,indx+self.seq_len)
                    instance = instances.iloc[indx:indx+self.seq_len].reset_index(drop = True)

                o,pred,_ = self.evaluate_set(instance,cons= True,kernel = kernel_mode)

            if type(instances) == type(np.array([])) or type(instances) == type([]):

                instance = pd.DataFrame(instances[indx], columns = self.input_names)
                o,pred,_ = self.evaluate_set(instance,cons= True,kernel = kernel_mode)

            pred_row = []
            for key in self.output_names:
                if key in pred.keys():
                    pred_row.append((pred[key]))
                else:
                    pred_row.append(0)
            pred_arr.append(pred_row)
            active_r = 0

            for j in o:
                if float(j[1]) != 0:
                    active_r+= 1
            active_arr.append(active_r)

        pred_arr = pd.DataFrame(np.array(pred_arr).reshape(len(instances)-self.seq_len+1,len(self.output_names)),columns = self.output_names)
        return pred_arr,active_arr

    def LearningClassifierSystem(self,instances,outputs,mutation_frq = 10,      # Train the learning classifer system
                                sub_thresh = 0.1,epochs = 50,eval = None,
                                adaptive_purge = None,verberose = True):
        self.history = {}
        self.history['metric'] = []
        self.history['fitness'] = []
        self.history['activations'] = []
        self.history['fitness_std'] = []
        self.history['rules'] = []
        time_arr = []
        start1 = timer()
        for i in range(epochs):
            if type(adaptive_purge) != type(None):
                self.purge_threshold = (adaptive_purge/0.67)*(1-2.71**(-i/range_s))

            start = timer()
            self.train(instances,outputs,mutation_freq = mutation_frq)
            self.GA.check_names(self.rules)
            end  = timer()

            time_arr.append(end-start)

            self.rules = self.subsume(self.rules,subsumtion_treshold = sub_thresh)

            fitness = []
            for rule in self.rules:
                fitness.append(rule.fitness)
            mean = np.mean(fitness)
            std = np.std(fitness)

            if self.type == "Continuous":

                if type(eval) != type(None):
                    pred,activations = self.evaluate_data(eval[0])
                    # print(pred.apply(np.ceil))
                    # print(eval[1].astype("int"))
                    self.history['metric'].append(mean_absolute_error(eval[1],pred))
                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ", mean," +/- ",std, " # of Rules:",len(fitness)," mae: ",mean_absolute_error(eval[1],pred)," ttc: ",(end1-start1)," sec")

                else:
                    pred,activations = self.evaluate_data(instances)
                    self.history['metric'].append(mean_absolute_error(outputs,pred))
                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ",  mean," +/- ",std, " # of Rules:",len(fitness)," mae: ",mean_absolute_error(outputs,pred)," ttc: ",(end1-start1)," sec")

            if self.type == "Multi-Class":

                if type(eval) != type(None):
                    pred,activations = self.evaluate_data(eval[0])

                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    # eval_arr = []
                    # pred_arr = []
                    # for row in range(len(pred)):
                    #     ee = np.argmax(eval[1].values[row])
                    #     pp = np.argmax(pred.values[row])
                    #     eval_arr.append(ee)
                    #     pred_arr.append(pp)
                    #print(eval[1].dtypes)
                    #print(pred.apply(np.ceil).dtypes)
                    # for rule in self.rules:
                    #     if rule.fitness >= 0.5:
                    #         print(rule)
                    self.history['metric'].append(accuracy_score(eval[1],pred.apply(np.ceil).astype("int").clip(0, 1)))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ", mean," +/- ",std, " # of Rules:",len(fitness)," acc: ",accuracy_score(eval[1],pred.apply(np.ceil).astype("int").clip(0, 1))," ttc: ",(end1-start1)," sec")

                else:
                    pred,activations = self.evaluate_data(instances)
                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    eval_arr = []
                    pred_arr = []
                    # for row in range(len(pred)):
                    #     ee = np.argmax(outputs.values[row])
                    #     pp = np.argmax(pred.values[row])
                    #     eval_arr.append(ee)
                    #     pred_arr.append(pp)

                    self.history['metric'].append(accuracy_score(outputs.astype('int'),pred.apply(np.ceil).astype('int')))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ",  mean," +/- ",std, " # of Rules:",len(fitness)," acc: ",accuracy_score(outputs.astype("int"),pred.apply(np.ceil).astype("int"))," ttc: ",(end1-start1)," sec")

            if self.type == "Class":

                if type(eval) != type(None):
                    pred,activations = self.evaluate_data(eval[0])
                    # print(pred)
                    # print(eval[1])
                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    eval_arr = []
                    pred_arr = []
                    for row in range(len(pred)):
                        ee = np.argmax(eval[1].values[row])
                        pp = np.argmax(pred.values[row])
                        eval_arr.append(ee)
                        pred_arr.append(pp)

                    self.history['metric'].append(accuracy_score(eval_arr,pred_arr))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ", mean," +/- ",std, " # of Rules:",len(fitness)," acc: ",accuracy_score(eval_arr,pred_arr)," ttc: ",(end1-start1)," sec")

                else:
                    pred,activations = self.evaluate_data(instances)
                    self.history['fitness'].append(mean)
                    self.history['fitness_std'].append(std)
                    self.history['rules'].append(len(self.rules))
                    eval_arr = []
                    pred_arr = []
                    for row in range(len(pred)):
                        ee = np.argmax(outputs.values[row])
                        pp = np.argmax(pred.values[row])
                        eval_arr.append(ee)
                        pred_arr.append(pp)

                    self.history['metric'].append(accuracy_score(eval_arr,pred_arr))
                    if not verberose:
                        print(str(i+1) +"/" +str(epochs) + " Epochs |"+str(len(self.rules)) +" Rules |"+str(np.mean(time_arr))+ " sec/epoch | Time left:",str(np.mean(time_arr)*((epochs-i)+1)),end='\r')
                    if verberose or i == epochs-1:
                        end1  = timer()
                        print("Epoch: ",i," Avg. Fitness: ",  mean," +/- ",std, " # of Rules:",len(fitness)," acc: ",accuracy_score(eval_arr,pred_arr)," ttc: ",(end1-start1)," sec")


    def train(self,instances,outputs,mutation_freq = 10):                       # train the classifer system
        #this need to be changed to np array which is internaly converted to df!
        mutation_cnt = 0


        for indx in range(len(instances)-(self.seq_len)):

            if type(instances) == type(pd.DataFrame()):
                if self.seq_len == 1:

                    instance = instances.iloc[[indx]].reset_index(drop = True)
                    output = outputs.iloc[[indx]].reset_index(drop = True)

                if self.seq_len > 1:

                    #print(len(instances)-(self.seq_len),indx,indx+self.seq_len)
                    instance = instances.iloc[indx:indx+self.seq_len].reset_index(drop = True)
                    output = outputs.iloc[[indx+self.seq_len]].reset_index(drop = True)
                self.GA.check_names(self.rules)

                matched,pred,tot_fit = self.match(instance,consequent = True)

            if type(instances) == type(np.array) or type(instances) == type([]):

                instance = instances[indx].reset_index(drop = True)
                output = outputs[indx].reset_index(drop = True)
                #print(instance,output)
                self.GA.check_names(self.rules)
            # for rule in self.rules:
            #     print(rule.name)
            matched,pred,tot_fit = self.match(instance,consequent = True)


            if len(matched) != 0:
                #print(matched)
                for rule in self.rules:

                    if rule.name in matched[:,0]:

                        index = np.where(matched[:,0] == rule.name)

                        #print(index,len(matched[index,1][0]))

                        rule_output = float(matched[index,1][0])

                        key = matched[index,2][0][0]

                        #print(key)

                        ensamble_out = float(pred[key])
                        ensamble_fit = float(tot_fit[key])

                        target = output[key].values.tolist()[0]

                        self.update_fitness(rule,target,rule_output,ensamble_out,ensamble_fit,len(matched))

                #self.subsume(self.rules)

                self.prune_rules(purge = True)

            if len(matched) < self.covering_threshold:
                #print(instance,output)
                self.covering(instance,output, kernel_in= instance.reindex(index=instance.index[::-1]).values.ravel())
            mutation_cnt +=1
            if mutation_cnt > mutation_freq:
                mutation_cnt = 0
                if len(self.rules) != 0:
                    self.rules = self.GA.BasicGA(self.rules)
                else:
                    self.covering(instance,output, kernel_in= instance.reindex(index=instance.index[::-1]).values.ravel(),pop  = 5)

            #print(self.rules)

    def update_fitness(self,rule,tar,out,en_out,tot_fit,how_many_rules):        # update rules fitness
        err = tar-out
        cluster = (how_many_rules-1)/2


        if en_out == float(np.nan):
            new_correctness = 0
        else:
            if self.GA.max_output_attributes > 0:
                new_correctness = np.maximum(0,1-np.abs(tar-out))*self.fitness_weights[0] + np.maximum(0,(1-(np.abs(cluster))))*self.fitness_weights[1] + 0.001+ np.maximum(0,(1-np.abs(tar-en_out)))*self.fitness_weights[2]
            else:
                if tar == 0:
                    new_correctness = 0
                else:
                    new_correctness = np.maximum(0,(1-(np.abs(cluster))))*self.fitness_weights[1] + 0.001
                    if np.abs(tar-out) <= 0.05:
                        new_correctness += 1*self.fitness_weights[0]
                    if np.abs(tar-en_out) <= 0.05:
                        new_correctness += 1*self.fitness_weights[2]
            #new_correctness/=3

        rule.matched +=1
        rule.correctness+=new_correctness
        rule.fitness = (rule.correctness)/rule.matched
        #print(rule.name,rule.fitness)


    def prune_rules(self,offset = 0, purge = False, purge_threshold = None,nursery = True): # purge bad rules form the population using fitness and match/eval ratio

        if type(purge_threshold) == type(None):
            purge_threshold = self.purge_threshold

        if purge:

            temp = []

            for rule in self.rules:

                if (rule.matched <= self.min_age and rule.evaluated <= 1/self.eval_match_ratio) or (rule.matched/rule.evaluated >= self.eval_match_ratio and rule.fitness > purge_threshold):
                    temp.append(rule)
                else:
                    self.discarder_rules.append(rule)

            self.rules = temp

        cntr = 0

        if nursery:
            for i in self.rules:
                if rule.matched <= self.min_age:
                    cntr+=1
        #print(cntr)

        if len(self.rules) > (self.max_pop+cntr) - offset and len(self.rules) > 1:
            rule_list = []

            for rule in self.rules:
                rule_list.append(np.array([rule.name,rule.fitness,rule.matched]))

            rule_list = np.array(rule_list)

            rule_list =rule_list[np.argsort(rule_list.reshape(len(self.rules),3)[:, 1]),:]

            #print(rule_list)

            new_rules = rule_list[-self.max_pop:,:]

            temp_rules = []

            # delete if pop too big
            for rule in self.rules:

                if rule.name in new_rules[:,0]:

                    temp_rules.append(rule)
                else:

                    self.discarder_rules.append(rule)

            self.rules = temp_rules


    def match(self,instance,consequent = False,p = None):                       # match rules to data

        matched,pred,tot_fit = self.evaluate_set(instance,cons = consequent)
        #print(matched,pred,tot_fit)
        return matched,pred,tot_fit


    def _evaluate_inner_loop(self,cnt_eval,rule,input,kernel_in,kernel,cons,matched_only):
                    df = []
                    if cnt_eval:
                        rule.evaluated+=1

                    # print(rule.name)
                    # print(input)

                    if len(rule.kernel) != 0 and kernel == True:
                        out = rule.evaluate(input,
                                            name_var = True,
                                            con = cons,
                                            kernel = True,
                                            kern_in = kernel_in)
                    else:
                        out = rule.evaluate(input,name_var = True,con = cons)


                    #print(out)
                    rule_name = rule.name
                    rule_fit = rule.fitness
                    rule_matches = rule.matched

                    mod = 1
                    if rule_matches <= self.min_age:
                        mod = 0

                    if out != "err0" and out != "err1":
                        if type(out) == type(True):
                            df.append([rule_name,out,None,rule_fit])
                        else:
                            df.append([rule_name,out[0],out[1],rule_fit*mod])
                    else:
                        if not matched_only:
                            df.append([rule_name,out,None,rule_fit])
                    #print(df)

                    return df

    def evaluate_set(self,input,                                                # evaluate specified data
                    next_data = {},
                    rules = None,
                    cons = False,
                    matched_only = True,
                    kernel = True,cnt_eval= True,kern_in= None):

        if rules == None:
            rules = self.rules

        df = []
        #print(input)
        if kernel:
            if type(kern_in) != type(None):
                kernel_in = kern_in
            else:
                kernel_in = input.reindex(index=input.index[::-1]).values.ravel()
                #print(kernel_in)
        else:
            kernel_in = None
        inp_arr= []

        for rule in rules:
            inp_arr.append([cnt_eval,rule,input,kernel_in,kernel,cons,matched_only])
        #print(inp_arr)
        #
        # #pool = multiprocessing.Pool(6)
        # # pool = None
        # # if type(pool) != type(None):
        # #     results = pool.starmap(self._evaluate_inner_loop, inp_arr)
        # #     pool.close()
        # #     pool.join()
        # #     for result in results:
        # #         #print(result)
        # #         for sample in result:
        # #             if len(df) != 0:
        # #                 df.append(sample
        #     #print(np.array(df))
        # else:
        for rule in rules:
            temp = self._evaluate_inner_loop(cnt_eval,rule,input,kernel_in,kernel,cons,matched_only)
            #print(temp)
            if len(temp) != 0:
                df.append(temp[0])
            #print(np.array(df))
        #print(df)

        predictions = {}
        total_fit_dict ={}
        df = np.array(np.array(df))
        if len(df) != 0:
            for key in self.output_names:
                temp_out = 0
                #print(df)
                array = df[df[:,2] == key]
                #print(array)
                total_fit = np.sum(df[:,3].astype(np.float))
                if total_fit ==0:
                    total_fit = 1
                total_fit_dict
                for row in array:
                    #print(row[3],total_fit)
                    temp_out  += (float(row[1])*(float(row[3]))/float(total_fit))
                predictions[key]=temp_out
                total_fit_dict[key] = total_fit
        #print(predictions)
        return df,predictions,total_fit_dict


    def get_rule(self,rule_name,rule_set):                                      # find rule by name (very useful)
        for i in rule_set:
            if i.name == rule_name:
                return i
        return None
