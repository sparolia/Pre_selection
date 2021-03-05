import numpy
import math
import sys
import ROOT
import random


def cal_pop_fitness(var, pop, tree_sig, tree_bkg,param_abs):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    #fitness = numpy.sum(pop*equation_inputs, axis=1)
    fitness = numpy.empty(len(pop))

    k_Err = ['abs(k_dxy/k_dxyErr)','abs(k_dz/k_dzErr)']
    #Filename_sig = sys.argv[1]
    #Filename_bkg = sys.argv[2]    

    for j in range(len(pop)):
        sig = 0
        bkg = 0 
        entries_sig = tree_sig.Filter('%s > %s'%(var[0],pop[j,0]))
        entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],pop[j,0]))
        for i in range(1,len(var)-param_abs):
            entries_sig = entries_sig.Filter('%s > %s'%(var[i],pop[j,i]))
            entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],pop[j,i]))
        for k in range(param_abs):
            entries_sig = entries_sig.Filter('%s < %s'%(k_Err[k],pop[j,k+i+1]))
            entries_bkg = entries_bkg.Filter('%s < %s'%(k_Err[k],pop[j,k+i+1]))
        sig_ent = entries_sig.Count();
        bkg_ent = entries_bkg.Count();
        sig = sig_ent.GetValue()
        bkg = bkg_ent.GetValue()
        if(bkg>0):
            fitness[j] = numpy.round((sig/(math.sqrt(bkg))),2)
            #fitness[j] = numpy.round((sig/(math.sqrt(bkg+sig))),2)
            #fitness[j] = numpy.round((2*((math.sqrt(bkg+sig))-(math.sqrt(bkg)))),2)
            #fitness[j] = numpy.round((math.sqrt((2*(sig+bkg))*(math.log(1+(sig+bkg)))-(2*sig))),2)
            #fitness[j] = numpy.round((sig/bkg),2)
        else:
            fitness[j] = 0
            
        #print (sig)
        
    return fitness


def cal_pop_fitness_final(var, pop, tree_sig, tree_bkg,param_abs):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    #fitness = numpy.sum(pop*equation_inputs, axis=1)
    fitness1 = numpy.empty(len(pop))
    fitness2 = numpy.empty(len(pop))
    fitness3 = numpy.empty(len(pop))
    fitness4 = numpy.empty(len(pop))
    fitness5 = numpy.empty(len(pop))
    k_Err = ['abs(k_dxy/k_dxyErr)','abs(k_dz/k_dzErr)']
    #Filename_sig = sys.argv[1]
    #Filename_bkg = sys.argv[2]    
    fom=[]
    #fom=numpy.empty(5)
    for j in range(len(pop)):
        sig = 0
        bkg = 0 
        entries_sig = tree_sig.Filter('%s > %s'%(var[0],pop[j,0]))
        entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],pop[j,0]))
        for i in range(1,len(var)-param_abs):
            entries_sig = entries_sig.Filter('%s > %s'%(var[i],pop[j,i]))
            entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],pop[j,i]))
        for k in range(param_abs):
            entries_sig = entries_sig.Filter('%s < %s'%(k_Err[k],pop[j,k+i+1]))
            entries_bkg = entries_bkg.Filter('%s < %s'%(k_Err[k],pop[j,k+i+1]))
        sig_ent = entries_sig.Count();
        bkg_ent = entries_bkg.Count();
        sig = sig_ent.GetValue()
        bkg = bkg_ent.GetValue()
        if(bkg>0):
            fitness1[j] = numpy.round((sig/(math.sqrt(bkg))),2)
            fitness2[j] = numpy.round((sig/(math.sqrt(bkg+sig))),2)
            fitness3[j] = numpy.round((2*((math.sqrt(bkg+sig))-(math.sqrt(bkg)))),2)
            fitness4[j] = numpy.round((math.sqrt((2*(sig+bkg))*(math.log(1+(sig+bkg)))-(2*sig))),2)
            fitness5[j] = numpy.round((sig/bkg),2)
        else:
            fitness1[j] = 0
            fitness2[j] = 0
            fitness3[j] = 0
            fitness4[j] = 0
            fitness5[j] = 0
        #print (sig)

    best_match_idx = numpy.where(fitness1 == numpy.max(fitness1))
    #print (best_match_idx[0])
    #fom('Figure of merits values:' +'\n')
    '''
    numpy.append(fom,fitness1)
    numpy.append(fom,fitness2)
    numpy.append(fom,fitness3)
    numpy.append(fom,fitness4)
    numpy.append(fom,fitness5)
    '''
    fom.append(fitness1)
    fom.append(fitness2)
    fom.append(fitness3)
    fom.append(fitness4)
    fom.append(fitness5)
    
    fom = numpy.array(fom)
    #print (fom)
    fom_best = (fom[:,best_match_idx[0]])


#fitness2[best_match_idx],fom,fitness3[best_match_idx],fitness4[best_match_idx],fitness5[best_match_idx]])
   
    return fitness1,fom_best

'''
            if(numpy.all(var_sig_1>pop[j,0:i+1]) and numpy.all(numpy.abs(var_sig_2) < pop[j,i+1:i+p+2])):
                sig +=1

        for entrybkg in range (0,tree_bkg.GetEntries()):
            tree_bkg.GetEntry(entrybkg)
            jpsi_mass=getattr (tree_bkg,"jpsi_mass")
            if( jpsi_mass < 3.05 or 3.15 < jpsi_mass):
                var_bkg_1 = []
                var_bkg_2 = []
                for k in range(len(equation_inputs)-param_abs):
                    var_bkg_1.append(getattr (tree_bkg,equation_inputs[k]))
                for q in range(param_abs):
                    var_bkg = (getattr (tree_bkg,equation_inputs[k+1+q]))
                    err_bkg = (getattr (tree_bkg,equation_inputs[k+1+q]+"Err"))
                    var_bkg_2.append(var_bkg/err_bkg)

                if(numpy.all(var_bkg_1>pop[j,0:k+1]) and numpy.all(numpy.abs(var_bkg_2) < pop[j,k+1:k+q+2])):
                    bkg +=1

        if(bkg>0):
            #print (bkg)
            fitness[j] = sig/(math.sqrt(bkg))
        else:
            fitness[j] = 0
        #print (sig)

    return fitness
'''



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
            gene_idx=numpy.random.randint(0,6)
            if(gene_idx == 4):
                random_value = round(random.uniform(-0.05, 0.05),2)
            else:
                random_value = round(random.uniform(-0.5, 0.5), 2)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
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
'''
