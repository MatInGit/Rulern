import numpy as np
import os, sys
import random
import copy
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .RuleModule import Rule
import clips

class ClipsOps():

    def __init__(self):

        self.rules = []
        self.inp_template = None
        self.out_template = None

        self.clips_sinp_template = None
        self.clips_out_template = None

        self.env = clips.Environment()
        #env.build
        return None

    def build_templates(self,sample_input):

        # build a rule template

        temp1 = "(deftemplate "
        temp_row = "data (slot index)"
        for column in sample_input:
            temp_row += " (slot "+column+")"
        temp_row += ")"
        temp1+= temp_row
        #print(temp1)
        self.env.build(temp1)

        temp2  = "(deftemplate prediction\n"
        temp_row = "\t(multislot out)"
        temp2 += temp_row + ")"
        #print(temp2)
        self.env.build(temp2)


    def assert_input(self,input):                                               # assert input
        for row in range(len(input)):
            template = self.env.find_template('data')
            new_fact = template.new_fact()
            new_fact['index'] = int(row - len(input)+1)
            for column in input.columns:
                #print(row,column)
                new_fact[column] = float(input.at[row,column])
            new_fact.assertit()

    def convert_rule(self,rule):                                                # convert rules from LCS to clips

        adj = rule.seq_len - 1
        new_rn= rule.name.replace('(', 'OB').replace(')', 'CB').replace('|', 'I')
        rule_str = "(defrule " + new_rn + "\n" + "\t(and\n"

        for row in range(rule.seq_len):
            temp_row = "\t(data (index " + "?"+"idxm"+str(adj - row)+")"
            for column in rule.df_kernel.columns:
                temp_row += " ("+column+ " ?"+ column+str(adj - row)+")"
            temp_row += ")\n"
            rule_str += temp_row

        for row in range(rule.seq_len):
            rule_str +="\t(test (eq "+"?"+"idxm"+str(adj-row)+" "+str(row-adj)+"))\n"

        for key in rule.ant:
            for index in range(len(rule.ant[key][0])):
                var = "?"+ key + str(rule.ant[key][0][index])
                op = rule.ant[key][1][index]
                if op == "==":
                    op = "eq"
                if type(rule.ant[key][2][index]) == type([]):
                    comp = "?"+ rule.ant[key][2][index][0] + str(rule.ant[key][2][index][1])
                else:
                    comp = str(rule.ant[key][2][index])
                rule_str +="\t(test ("+op+" "+ var +" "+comp+"))\n"
        rule_str += "\t)\n\t=>\n"

        if "vars" in rule.con.keys():
            predition_str = "\t(bind ?answer (+"
            for key in rule.con["vars"]:
                for indx in range(len(rule.con["vars"][key][0])):
                    calc_str = " (* "
                    if len(rule.con["vars"][key][0]) != 0:
                        #print(rule.con["vars"][key][0][indx])
                        calc_str += str(float(rule.con["vars"][key][1][indx]))
                        calc_str += " ?"+ key+ str(rule.con["vars"][key][0][indx])
                        calc_str += ") "
                    predition_str += calc_str
            predition_str += str(rule.con["bias"]) +"))\n"

        else:
            predition_str = "\t(bind ?answer (+ 0 "+ str(rule.con["bias"]) +"))\n"
        rule_str += predition_str
        rule_str += "\t(bind ?fit "+str(rule.fitness) +")\n"
        rule_str += "\t(bind ?rn "+ new_rn  +" )\n"
        rule_str += "\t(bind ?var "+rule.con["out_var"] +")\n"

        assertion_str = "\t(assert "
        assertion_str += " (out ?rn  ?var ?answer ?fit)"
        # assertion_str += " (fitness ?fit)"
        # assertion_str += " (out_var ?var)"
        # assertion_str += " (out_val ?answer)"
        assertion_str += ")\n"
        rule_str += assertion_str
        rule_str+= ")" #\t(printout t ?answer crlf)
        #print(rule_str)
        self.env.build(rule_str)



    def convert_rules(self,rules):
        return None


    def evaluate_rule(self):
        return None


    def evaluate_rules(self,input,rules = None):

        if rules == None:
            rules = self.rules

        env.clear()
        self.convert_data(input)


        return None
