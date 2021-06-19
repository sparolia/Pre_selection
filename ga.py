import numpy
import math
import sys
import ROOT
import random


def cal_pop_fitness(var, pop, tree_sig, tree_bkg, param_abs):
    #ROOT.ROOT.EnableImplicitMT() 
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    
    fitness = numpy.empty(len(pop))
    tot_tau = (tree_sig.Filter ("is_signal_channel > 0.5").Count()).GetValue()
    tot_muon = (tree_sig.Filter ("is_signal_channel < 0.5").Count()).GetValue()
    for j in range(len(pop)):
        #sig = 0
        #bkg = 0
        #F1 = 0.0
        #F2 = 0.0
        entries_sig = tree_sig.Filter('%s > %s'%(var[0],pop[j,0]))
        entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],pop[j,0]))
        for i in range(1,len(var)-(param_abs+1)):
            entries_sig = entries_sig.Filter('%s > %s'%(var[i],pop[j,i]))
            entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],pop[j,i]))
        #print (i)
        for k in range(param_abs):
            entries_sig = entries_sig.Filter('%s < %s'%(var[k+i+1],pop[j,k+i+1]))
            entries_bkg = entries_bkg.Filter('%s < %s'%(var[k+i+1],pop[j,k+i+1]))
        #print (k)
        entries_sig = entries_sig.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],pop[j,k+i+2],var[k+i+2],pop[j,k+i+3]))
        entries_bkg = entries_bkg.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],pop[j,k+i+2],var[k+i+2],pop[j,k+i+3]))

        cut_tau = (entries_sig.Filter ("is_signal_channel > 0.5").Count()).GetValue()
        cut_muon = (entries_sig.Filter ("is_signal_channel < 0.5").Count()).GetValue()
      
        #sig_evnt = entries_sig.Filter("is_signal_channel > 0.5")
        bkg_evnt_1 = entries_bkg.Filter("norm_weight < 1")
        bkg_evnt_2 = entries_bkg.Filter("norm_weight > 1 and norm_weight < 8")
        bkg_evnt_3 = entries_bkg.Filter("norm_weight > 8")
        
        sig_weight = round((entries_sig.Mean("norm_weight").GetValue()),3)
        bkg_weight_1 = round((bkg_evnt_1.Mean("norm_weight").GetValue()),3)
        bkg_weight_2 = round((bkg_evnt_2.Mean("norm_weight").GetValue()),3)
        bkg_weight_3 = round((bkg_evnt_3.Mean("norm_weight").GetValue()),3)

        sig = sig_weight*(entries_sig.Count().GetValue())
        #bkg = (entries_bkg.Count()).GetValue()
        bkg1 = bkg_weight_1*((bkg_evnt_1.Count()).GetValue())
        bkg2 = bkg_weight_2*((bkg_evnt_2.Count()).GetValue())
        bkg3 = bkg_weight_3*((bkg_evnt_3.Count()).GetValue())
        bkg = bkg1 + bkg2 + bkg3       
    
        tau = sig_weight*cut_tau

        eff_tau = cut_tau/tot_tau
        eff_muon = cut_muon/tot_muon
        lam = [0.25, 0.50, 0.75]

        if(bkg>0 and eff_muon >0):
            F1 = (tau/(math.sqrt(bkg)))                                                                    
            F2 = abs( 1-(eff_tau/eff_muon))
            fitness[j] = round(((F1*lam[2])+ ((1-lam[2])/F2)),3)
        else:
            fitness[j] = -1
        
    return fitness


def cal_pop_fitness_final(var, pop, tree_sig, tree_bkg,param_abs):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    
    fitness = numpy.empty(len(pop))
    fom=[]
    
    tot_tau = (tree_sig.Filter ("is_signal_channel > 0.5").Count()).GetValue()
    tot_muon = (tree_sig.Filter ("is_signal_channel < 0.5").Count()).GetValue()

    for j in range(len(pop)):
        
        entries_sig = tree_sig.Filter('%s > %s'%(var[0],pop[j,0]))
        entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],pop[j,0]))
        for i in range(1,len(var)-(param_abs+1)):
            entries_sig = entries_sig.Filter('%s > %s'%(var[i],pop[j,i]))
            entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],pop[j,i]))
        for k in range(param_abs):
            entries_sig = entries_sig.Filter('%s < %s'%(var[k+i+1],pop[j,k+i+1]))
            entries_bkg = entries_bkg.Filter('%s < %s'%(var[k+i+1],pop[j,k+i+1]))
        entries_sig = entries_sig.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],pop[j,k+i+2],var[k+i+2],pop[j,k+i+3]))
        entries_bkg = entries_bkg.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],pop[j,k+i+2],var[k+i+2],pop[j,k+i+3]))      
       
        cut_tau = (entries_sig.Filter ("is_signal_channel > 0.5").Count()).GetValue()
        cut_muon = (entries_sig.Filter ("is_signal_channel < 0.5").Count()).GetValue()

        #sig_evnt = entries_sig.Filter("is_signal_channel > 0.5")
        bkg_evnt_1 = entries_bkg.Filter("norm_weight < 1")
        bkg_evnt_2 = entries_bkg.Filter("norm_weight > 1 and norm_weight < 8")
        bkg_evnt_3 = entries_bkg.Filter("norm_weight > 8")
        
        sig_weight = round((entries_sig.Mean("norm_weight").GetValue()),3)
        bkg_weight_1 = round((bkg_evnt_1.Mean("norm_weight").GetValue()),3)
        bkg_weight_2 = round((bkg_evnt_2.Mean("norm_weight").GetValue()),3)
        bkg_weight_3 = round((bkg_evnt_3.Mean("norm_weight").GetValue()),3)

        sig = sig_weight*(entries_sig.Count().GetValue())
        #bkg = (entries_bkg.Count()).GetValue()
        bkg1 = bkg_weight_1*((bkg_evnt_1.Count()).GetValue())
        bkg2 = bkg_weight_2*((bkg_evnt_2.Count()).GetValue())
        bkg3 = bkg_weight_3*((bkg_evnt_3.Count()).GetValue())
        bkg = bkg1 + bkg2 + bkg3 
        
        tau = sig_weight*cut_tau

        #sig.append(sig_ent)
        #bkg.append(bkg_ent)
        eff_tau = (cut_tau/tot_tau)                                                                    
        eff_muon = (cut_muon/tot_muon)                                                                 
        lam = [0.25, 0.50, 0.75] 

        if(bkg > 0 and eff_muon > 0):
            F1 = tau/(math.sqrt(bkg))                                                     
            F2 = (abs(1-(eff_tau/eff_muon)))
            fitness[j] = round(((F1*lam[2])+((1-lam[2])/F2)),3)
        else:
            fitness[j] = -1
        #print (sig)

    #best_match_idx = (numpy.where(fitness == numpy.amax(fitness)))
    best_match_idx = numpy.argmax(fitness)
    '''    
    sig_best = (sig[best_match_idx])
    bkg_best = (bkg[best_match_idx])
    F1_best = F1[best_match_idx]
    F2_best = F2[best_match_idx]
    eff_tau_best = eff_tau[best_match_idx]
    eff_muon_best = eff_muon[best_match_idx]
    '''
    solution = (pop[best_match_idx,:])
    
    return fitness, solution#,fom_best


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    #mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        #gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            #random_value = numpy.random.uniform(-1.0, 1.0, 1)
            gene_idx=numpy.random.randint(0,13)
            if(gene_idx == 4 or gene_idx == 6):
                random_value = (random.uniform(-0.005, 0.005))
            elif(gene_idx == 5 or gene_idx == 7 or gene_idx == 11 or gene_idx == 12):
                random_value = (random.uniform(-0.0005, 0.0005))
            elif(gene_idx == 10):
                random_value = (random.uniform(-0.05, 0.05))
            else:
                random_value = (random.uniform(-0.50, 0.50))
            if(abs(random_value) < (0.1*offspring_crossover[idx, gene_idx])):
                offspring_crossover[idx, gene_idx] = ((offspring_crossover[idx, gene_idx] + random_value))
            else:
                offspring_crossover[idx, gene_idx] = ((offspring_crossover[idx, gene_idx] + (0.5*random_value)))
            #gene_idx = gene_idx + mutations_counter
    return offspring_crossover
'''
def mutation(offspring_crossover, num_mutations=1):

    # Mutation changes a single gene in each offspring randomly.

    for idx in range(offspring_crossover.shape[0]):

        # The random value to be added to the gene.

        random_value = numpy.random.uniform(-1.0, 1.0, 1)

        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value

    return offspring_crossover


def validation(var, cuts, tree_sig, tree_bkg, param_abs):
    
    #calculating the efficiency of cuts
    denom_sig = tree_sig.Count();
    denom_bkg = tree_bkg.Count();
    entries_sig = tree_sig.Filter('%s > %s'%(var[0],cuts[0]))
    entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],cuts[0]))
    for i in range(1,len(var)-param_abs):
        entries_sig = entries_sig.Filter('%s > %s'%(var[i],cuts[i]))
        entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],cuts[i]))
    for k in range(param_abs):
        entries_sig = entries_sig.Filter('%s < %s'%(var[k+i+1],cuts[j,k+i+1]))
        entries_bkg = entries_bkg.Filter('%s < %s'%(var[k+i+1],cuts[j,k+i+1]))
    sig_ent = entries_sig.Count();
    bkg_ent = entries_bkg.Count();
    sig = sig_ent.GetValue()
    bkg = bkg_ent.GetValue()

    eff_sig = sig/(denom_sig.GetValue())
    eff_bkg = bkg/(denom_bkg.GetValue())
    
    return eff_sig, eff_bkg
'''
