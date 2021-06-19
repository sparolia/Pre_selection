import numpy
import ga
import math
import sys
import ROOT
import matplotlib.pyplot

"""
The y=target is to maximize the Figure of Merit ASAP:
We are going to use the genetic algorithm for the best possible values after a number of generations.
"""
#ROOT.ROOT.EnableImplicitMT()

# Inputs of the equation.

#equation_inputs = [4,-2,3.5,5,-11,-4.7]
equation_inputs = ["mu1pt","mu2pt","kpt","jpsi_pt","jpsivtx_svprob","jpsivtx_cos2D","bvtx_svprob","bvtx_cos2D","Bpt_reco","Bmass","DR_jpsimu","ip3d"]


# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)+1
#print (num_weights)
"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""

tree_sig = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Sig.root')
tree_bkg = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Bkg.root')

sols_pop =  sys.argv[1]
gen =  sys.argv[2]
numb = (sys.argv[3])

sol_per_pop = int(sols_pop)
num_parents_mating = 20 #int(sol_per_pop/2)

#pre_cuts = [3.0, 3.0, 2.5, 6.0, 0.01, 15.0, 0.995]

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
low_lim = numpy.array([3.5, 3.0, 2.0, 6.0, 0.001, 0.992, 0.001, 0.992, 12.0, 3.7, 0.6, -0.008, 0.007])
high_lim = numpy.array([6.0, 5.0, 5.0, 11.0, 0.01, 0.997, 0.01, 0.997, 16.0, 4.2, 0.8, -0.005, 0.01])
#low_lim = numpy.array([[3.5, 3.0, 2.0, 6.0, 0.01, 12, 0.995],[3.5, 3.0, 2.0, 6.0, 0.01, 12, 0.995],[3.5, 3.0, 2.0, 6.0, 0.01, 12, 0.995],[3.0, 3.0, 2.0, 6.0, 0.01, 10, 0.995]])
#high_lim = numpy.array([[6.0, 5.0, 5.0, 11.0, 0.1, 18, 0.997],[5.0, 4.0, 4.0, 10.0, 0.1, 16, 0.997],[5.5, 4.5, 4.0, 11.0, 0.1, 16, 0.997],[5.5, 4.5, 3.0, 10.0, 0.1, 16, 0.997]])

param_abs = 1
num_generations = int(gen)

for n in range(int(numb)):
    outfile = "output_" + sols_pop + "_" + gen + "_run" +str(n)+ ".dat"
    f = open(outfile,'w')

    print ("opened file" + outfile)
    best_outputs = []

    mu1pt = numpy.random.uniform(low=low_lim[0], high=high_lim[0], size=sol_per_pop)
    mu2pt = numpy.random.uniform(low=low_lim[1], high=high_lim[1], size=sol_per_pop)
    kpt = numpy.random.uniform(low=low_lim[2], high=high_lim[2], size=sol_per_pop)
    jpsi_pt = numpy.random.uniform(low=low_lim[3], high= high_lim[3], size=sol_per_pop)
    jpsivtx_svprob = numpy.random.uniform(low=low_lim[4], high=high_lim[4], size=sol_per_pop)
    jpsivtx_cos2D = numpy.random.uniform(low=low_lim[5], high=high_lim[5], size=sol_per_pop)
    bvtx_svprob = numpy.random.uniform(low=low_lim[6], high=high_lim[6], size=sol_per_pop)
    bvtx_cos2D = numpy.random.uniform(low=low_lim[7], high=high_lim[7], size=sol_per_pop)
    Bpt_reco = numpy.random.uniform(low=low_lim[8], high= high_lim[8], size=sol_per_pop)
    Bmass = numpy.random.uniform(low=low_lim[9], high= high_lim[9], size=sol_per_pop)
    DR_jpsimu = numpy.random.uniform(low=low_lim[10], high= high_lim[10], size=sol_per_pop)
    ip3d_ns = numpy.random.uniform(low=low_lim[11], high= high_lim[11], size=sol_per_pop)
    ip3d_ps = numpy.random.uniform(low=low_lim[12], high= high_lim[12], size=sol_per_pop)

    new_population = (numpy.vstack((mu1pt,mu2pt,kpt,jpsi_pt,jpsivtx_svprob,jpsivtx_cos2D,bvtx_svprob,bvtx_cos2D,Bpt_reco,Bmass,DR_jpsimu,ip3d_ns,ip3d_ps)).T)
    #print (new_population)
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        
        fitness = ga.cal_pop_fitness(equation_inputs, new_population, tree_sig, tree_bkg, param_abs)
        #best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        best_outputs.append(numpy.max(fitness))
        # The best result in the current iteration.
        #print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        #print(numpy.max(fitness))
        #f.write(fitness+"\n")
        
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
        #print("Parents")
        #print(parents)
        
        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        #print("Crossover")
        #print(offspring_crossover)
        
        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=num_weights)
        #print("Mutation")
        #print(offspring_mutation)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    (fitness,solution) = ga.cal_pop_fitness_final(equation_inputs, new_population, tree_sig, tree_bkg,param_abs)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))
    #indx = int(best_match_idx[0])
    Best_solution = numpy.asarray(new_population[best_match_idx[0], :])
    #print(Best_solution[0])
    #(eff_sig_new,eff_bkg_new) = ga.validation(equation_inputs,Best_solution[0],tree_sig_val, tree_bkg_val, param_abs) 
    #(eff_sig_pre,eff_bkg_pre) = ga.validation(equation_inputs,pre_cuts,tree_sig_val, tree_bkg_val, param_abs) 

    #f.write("Signal events:" + str(sig)+'\n')
    #f.write("Background events:" + str(bkg)+'\n')
    #f.write("Best figure of merits:"+ str(fom)+'\n')
    f.write("Best_outputs:" + str(best_outputs)+ "\n")
    #f.write("Best solution : " + str(new_population[best_match_idx, :])+ "\n")
    #f.write("Best solution : " + str(Best_solution)+'\n')
    f.write("Best solution : " + str(solution)+'\n')
    #f.write("Best fitness_1 : " + str(F1)+'\n')
    #f.write("Best fitness_2 : " + str(F2)+'\n')
    f.write("Best solution fitness : " + str(fitness[best_match_idx])+'\n')
    #f.write("Efficiency of Signal events after optimization: "+ str(eff_sig_new)+'\n')
    #f.write("Efficiency of Background events after optimization : "+ str(eff_bkg_new)+'\n')
    #f.write("Efficiency of Signal events for predefined cuts: "+ str(eff_sig_pre)+'\n')
    #f.write("Efficiency of Background events for predefined cuts : "+ str(eff_bkg_pre)+'\n')

    numpy.save('solution_%s_%s_%s'%(sols_pop,gen,n),solution)
    print ("saved file" + outfile)

    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    #matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("fitness_" + sols_pop + "_" + gen + "_run" + str(n) + ".png")

    print ("saved fitness plot" + str(n)) 
