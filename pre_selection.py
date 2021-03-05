import numpy
import ga
import math
import sys
import ROOT
import matplotlib.pyplot

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.

#equation_inputs = [4,-2,3.5,5,-11,-4.7]
equation_inputs = ["mu1pt","mu2pt","kpt","jpsi_pt","jpsivtx_svprob","k_dxy","k_dz"]


# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sols_pop =  sys.argv[1]
gen =  sys.argv[2]

sol_per_pop = int(sols_pop)
num_parents_mating = 15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #15 #int(sol_per_pop/2)

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
mu1pt = numpy.random.uniform(low=2.0, high=6.0, size=sol_per_pop)
mu2pt = numpy.random.uniform(low=2.0, high=6.0, size=sol_per_pop)
kpt = numpy.random.uniform(low=2.0, high=6.0, size=sol_per_pop)
jpsi_pt = numpy.random.uniform(low=0.0, high=10.0, size=sol_per_pop)
#bvtx_svprob = numpy.random.uniform(low=0.0, high=0.2, size=sol_per_pop)
jpsivtx_svprob = numpy.random.uniform(low=0.01, high=0.1, size=sol_per_pop)
k_dxy = numpy.random.uniform(low=4.0, high=6.0, size=sol_per_pop)
k_dz = numpy.random.uniform(low=4.0, high=6.0, size=sol_per_pop)

param_abs = 2

#new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
#new_population = (numpy.vstack((mu1pt,mu2pt,kpt,jpsi_pt,jpsivtx_svprob,k_dxy,k_dz)).T)
#print(new_population)

"""
new_population[0, :] = [2.4,  0.7, 8, -2,   5,   1.1]
new_population[1, :] = [-0.4, 2.7, 5, -1,   7,   0.1]
new_population[2, :] = [-1,   2,   2, -3,   2,   0.9]
new_population[3, :] = [4,    7,   12, 6.1, 1.4, -4]
new_population[4, :] = [3.1,  4,   0,  2.4, 4.8,  0]
new_population[5, :] = [-2,   3,   -7, 6,   3,    3]

"""
#Filename_sig = "root://cms-xrd-global.cern.ch//store/user/garamire/ntuples/2021Jan29/BcToJpsiTauNu_UL_2021Jan26.root"
Filename_sig = "/scratch/parolia/2021Jan29/BcToJpsiTauNu_UL_2021Jan26.root"
Filename_bkg = "/scratch/parolia/2021Jan29/data_UL_2021Jan29.root"

#File_sig = ROOT.TFile.Open(Filename_sig,"READ")
#File_bkg = ROOT.TFile.Open(Filename_bkg,"READ")
tree_sig_raw = ROOT.RDataFrame("BTo3Mu",Filename_sig)
tree_data = ROOT.RDataFrame("BTo3Mu",Filename_bkg)
#index = (len(pop))
tree_bkg_raw = tree_data.Filter('jpsi_mass < 2.97 or 3.20 < jpsi_mass')
entries1 = tree_sig_raw.Count();
entries2 = tree_bkg_raw.Count();
#print (entries1.GetValue())
#print (entries2.GetValue())
'''
test1_sig = tree_sig.Filter('abs(mu1eta) < 1.2 && abs(mu2eta) < 1.2','cut1_sig')
test1_bkg = tree_bkg.Filter('abs(mu1eta) < 1.2 && abs(mu2eta) < 1.2','cut1_bkg')
test2_sig = tree_sig.Filter('(abs(mu1eta) < 1.2 && (1.2 < abs(mu2eta) and abs(mu2eta) < 2.4)) or (abs(mu2eta) < 1.2 && (1.2 < abs(mu1eta) and abs(mu1eta) < 2.4))','cut2_sig')
test2_bkg = tree_bkg.Filter('(abs(mu1eta) < 1.2 && (1.2 < abs(mu2eta) and abs(mu2eta) < 2.4)) or (abs(mu2eta) < 1.2 && (1.2 < abs(mu1eta) and abs(mu1eta) < 2.4))','cut2_bkg')

test3_sig = tree_sig.Filter('abs(mu1eta) > 1.2 && abs(mu2eta) > 1.2','cut3_sig')
test3_bkg = tree_bkg.Filter('abs(mu1eta) > 1.2 && abs(mu2eta) > 1.2','cut3_bkg')

allReport_sig = tree_sig.Report()
allReport_bkg = tree_bkg.Report()
allReport_sig.Print()
allReport_bkg.Print()
'''
cut = []
cut.append('abs(mu1eta) < 1.2 && abs(mu2eta) < 1.2')
cut.append('abs(mu1eta) < 1.2 && (1.2 < abs(mu2eta) and abs(mu2eta) < 2.4)')
cut.append('abs(mu2eta) < 1.2 && (1.2 < abs(mu1eta) and abs(mu1eta) < 2.4)')
cut.append('(1.2 < abs(mu1eta) and abs(mu1eta) < 2.4) && (1.2 < abs(mu2eta) and abs(mu2eta) < 2.4)')

num_generations = int(gen)

outfile = "output_" + sols_pop + "_" + gen + ".dat"
f = open(outfile,'w')

for i in range(4):
    f.write(str(i) + "\n")
    best_outputs = []
    tree_sig = tree_sig_raw.Filter(cut[i],'cut_sig_%s'%i)
    tree_bkg = tree_bkg_raw.Filter(cut[i],'cut_bkg_%s'%i)
    new_population = numpy.round((numpy.vstack((mu1pt,mu2pt,kpt,jpsi_pt,jpsivtx_svprob,k_dxy,k_dz)).T),2)
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        #name_sig = 'test'+str(i)+'_sig'
        #name_bkg = 'test'+str(i)+'_bkg'
        fitness = ga.cal_pop_fitness(equation_inputs, new_population, tree_sig, tree_bkg,param_abs)
        #print("Fitness")
        #print(fitness)
        #f.write(fitness+"\n")
        #best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        best_outputs.append(numpy.max(fitness))
        # The best result in the current iteration.
        #print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    
        #f.write(fitness+"\n")

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
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
    (fitness,fom) = ga.cal_pop_fitness_final(equation_inputs, new_population, tree_sig, tree_bkg,param_abs)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))


    f.write("Best figure of merits:"+ str(fom)+'\n')
    f.write("Best_outputs:" + str(best_outputs)+ "\n")
    f.write("Best solution : " + str(new_population[best_match_idx, :])+ "\n")
    f.write("Best solution fitness : " + str(fitness[best_match_idx])+'\n')
    #print("Best solution fitness : ", fitness[best_match_idx])

    
    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    #matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("fitness_"+ str(i)+'_' + sols_pop + "_" + gen + ".png")

    allReport_sig = tree_sig.Report()
    allReport_bkg = tree_bkg.Report()
    #f.write(str(allReport_sig)+'\n')
    #f.write(str(allReport_bkg)+'\n')
    #allReport_sig.Print()
    #allReport_bkg.Print()

#File_sig.Close()
#File_bkg.Close()
f.close()
