from numbers import Real
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import cross_val_score
import numpy as np
import scipy as sp

class BoolChromosome():

    def __init__(self, *, size=-1, parents:list=None):
        if (size<=0 and parents==None) or (size>0 and parents is not None):
            raise ValueError("None paramiter received")
        
        self.qualitySingle = False

        if (size > 0):
            self.size = size
            allFalse = True
            while allFalse: # It MUST be at least one gene at True
                self.genes = np.random.choice([True, False], size=self.size)
                allFalse = self.genes.sum() == 0
            self.genes = self.genes.reshape((-1,self.genes.size))
            return
        
        self.size = None
        parents_matrix = None
        # check that the list of parents are chromosomes
        # and create the matrix of all the parents genes
        for parent in parents:
            if type(parent) is not BoolChromosome:
                raise ValueError("A parent in the list is not a Chromosome")
            # check that all the parents have the same size
            if self.size == None:
                self.size = parent.size
            if self.size != parent.size:
                raise ValueError("Found parents with different size")
            # create a matrix of parents genes
            if type(parents_matrix) is not np.ndarray:
                parents_matrix  = parent.genes
            else:
                parents_matrix = np.concatenate([parents_matrix, parent.genes], axis=0)

        # create a 1D array in which each value is from which parent take the gene
        allFalse = True
        while allFalse:
            allEqual = True
            while allEqual:
                random_mix = np.random.default_rng().integers(len(parents), size=self.size)
                allEqual = np.unique(random_mix).size == 1
            self.genes = np.diag(parents_matrix[random_mix, :])
            allFalse = self.genes.sum() == 0
        self.genes = self.genes.reshape((-1,self.genes.size))


    def __str__(self) -> str:
        return np.array2string(self.genes)
    
    def mutate(self, threshold=0.5, differentFrom=None):
        allFalse = True

        if differentFrom != None:
            inSet = True
        else:
            inSet = False

        while allFalse or inSet: # avoid to have all to zero
            mask = np.random.random(size=self.size) < threshold
            # mutate the single gene in the chromosome with the probability given by p_gene in the mask
            mutation = np.random.choice([True, False], size=self.size)
            self.genes = self.genes ^ (mutation & mask)
            allFalse = self.genes.sum() == 0

            if differentFrom != None:
                inSet = self in differentFrom
            else:
                inSet = False

        return self

    def __eq__(self, other) -> bool:
        if type(other) is BoolChromosome:
            return np.array_equal(self.genes, other.genes)
        return False
    
    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def getQuality(self, X=None, y=None, estimator=None, scoring="roc_auc", cv=5, n_jobs=-1):
        if self.qualitySingle is True:
            return self.quality
        
        X_new = np.array(X)[:, self.genes.ravel()]
        self.quality = cross_val_score(
            estimator,
            X_new,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        ).mean()
        self.qualitySingle = True
        return self.quality
    
    def nPositive(self) -> int:
        return int(self.genes.sum())
    
    def __hash__(self) -> int:
        return self.nPositive()
    
    def distance(self, other) -> int:
        # return the Hamming distance
        if type(other) is BoolChromosome:
            return (self.genes!=other.genes).sum()
        raise ValueError("You are trying to compute the distance from another class")

    def diversity(self, others:list) -> int:
        # The diversity from all the others: diversity is 1/d^2
        # we do the sum of all the diversities from the others list
        totDiv = 0
        for other in others:
            if type(other) is not BoolChromosome:
                raise ValueError("You are trying to compute the diversity from another class")
            if other != self:
                totDiv += 1/self.distance(other)**2
        return totDiv
    
    def copy(self):
        copy = BoolChromosome(size=self.size)
        copy.genes = self.genes
        return copy

class GeneticSelection(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    '''
    This Class was never completed. Therefore could not work...
    '''
    
    def __init__(self, estimator, *, pop_size=1000, scoring="roc_auc", cv=5, n_jobs=-1, verbose=False):

        self.pop_size = pop_size
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fitness(self):
        # Standard method
        totQuality = 0
        for _, (q, _) in self.population.items():
            totQuality += q
        for c, (q, _) in self.population.items():
            f = q/totQuality
            self.population[c] = (c, f)
        return
    
    def __rankMethod(self, p):
        # THE POPULATION MUST BE ORDERED BY THE RANK THAT YOU WANT
        i=0
        for c,(q, _) in self.population.items():
            if i < len(self.population)-1:
                f = (1-p)**i * p
            else:
                f = (1-p)**i
            self.population[c] = (q, f)
            i += 1

    def fitness(self, p=2/3):
        # Rank space method
        self.population = {k: v for k, v  in sorted(self.population.items(), key=lambda item: item[1][0], reverse=True)} # Sort by quality

        self.__rankMethod(p)

    def fitness(self, p=2/3):
        # Order the population by Diveristy
        chromosomes = np.array([c for c in self.population.keys()])
        diversity = [c.diversity(chromosomes) for c in self.population.keys()]
        chromosomes = chromosomes[np.argsort(diversity)] # Sorting by the diversity

        # Computing ranks
        diversity_rank = np.arange(chromosomes.size) + 1 # Since the chromosome are ordered the diversity rank is [1, 2, 3, ...]
        quality = np.array([c.getQuality() for c in chromosomes])
        quality_rank = sp.stats.rankdata(-quality, method="ordinal") # - quality because the reverse order

        # Union of the ranks
        sum_rank = quality_rank + diversity_rank 
        comb_rank = sp.stats.rankdata(sum_rank, method="ordinal") # Since we have ordered at the begin by diveristy, in case of same sum, is choosed the one with the best diveristy.

        # Sort by the union of ranks
        chromosomes = np.array(chromosomes)[np.argsort(comb_rank)]
        self.population = {c: (c.getQuality(), -1) for c in chromosomes}

        self.__rankMethod(p)

    def fitness(self, p=2/3):
        # Order the population by diveristy and the by number of positivies Features
        # In this way, for the individuals with same number of positive features is chosen before the one with more diveristy
        chromosomes = np.array([c for c in self.population.keys()])
        # diversity = [c.diversity(chromosomes) for c in self.population.keys()]
        # chromosomes = chromosomes[np.argsort(diversity)] # Sorting by the diversity
        nPos = [c.nPositive() for c in chromosomes]
        chromosomes = chromosomes[np.argsort(nPos)] # Sorting by the number of Positive

        # Computing ranks
        nPos_rank = np.arange(chromosomes.size) + 1 # Since the chromosome are ordered the number of Pos features rank is [1, 2, 3, ...]
        diversity = [c.diversity(chromosomes) for c in self.population.keys()]
        diveristy_rank = sp.stats.rankdata(diversity, method="ordinal")
        quality = np.array([c.getQuality() for c in chromosomes])
        quality_rank = sp.stats.rankdata(-quality, method="ordinal") # - quality because the reverse order

        # Union of the ranks
        sum_rank = 0.4*nPos_rank + 0.2*diveristy_rank + 0.4*quality_rank
        comb_rank = sp.stats.rankdata(sum_rank, method="ordinal") # Since we have ordered at the begin by nPos, in case of same sum, is choosed the one with the lower number of True elements.

        # Sort by the union of ranks
        chromosomes = np.array(chromosomes)[np.argsort(comb_rank)]
        self.population = {c: (c.getQuality(), -1) for c in chromosomes}

        self.__rankMethod(p)

    def fit(self, X, y=None):

        self.size = X.shape[1] # number of columns
        self.population = {} # {chromosome, (quality, fitness)}
        self.allGen = set()

        for _ in range(self.pop_size):
            chromosome:BoolChromosome = BoolChromosome(size=self.size)
            self.population[chromosome] = (chromosome.getQuality(X, y, self.estimator, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs), -1)
            self.allGen.add(chromosome)

        gen = 0
        r = 0
        best : BoolChromosome = None
        while r<=5:
            # compute the fitness for each individual
            self.fitness()
            # sort the population by fitness and resize to the population size 
            self.population = [k for k, _ in sorted(self.population.items(), key=lambda indv: indv[1][1], reverse=True)][:self.pop_size]

            print("Size %d" % len(self.population))
            print("Generation %d\tScore %f\t" % (gen, self.population[0].getQuality())) 
            if self.verbose:
                print("Selected Features")
                print(self.population[0])
            print("Number of selected features")
            print(self.population[0].nPositive())


            if best is not None:
                if best == self.population[0]:
                    r += 1
                else:
                    r = 0

            # the new generation
            new_generation = {}
            if self.verbose:
                print("Best of the old")
                print("-------------------------")
            
            # the new generation is composed by a part of the old people that survived in the old generation. [The VETERANS]
            s = int(0.20*self.pop_size)
            i = 0
            for chromosome in self.population:
                if i == s:
                    break
                if self.verbose:
                    print(chromosome, "score", chromosome.getQuality())
                new_generation[chromosome] = (chromosome.getQuality(), -1)
                i += 1

            # During their life there was a mutation of the genes
            mutated_population = [individual.copy().mutate(differentFrom=self.allGen) for individual in self.population]

            if self.verbose:
                print("Mutation of the best")
            # the new generation is composed also by the old people that survived and that in their life they made a mutation of the genses [The VETERANS mutated]
            s = int(0.20*self.pop_size)
            i = 0
            for chromosome in mutated_population:
                if i == s:
                    break
                new_generation[chromosome] = (chromosome.getQuality(X, y, self.estimator, scoring=self.scoring,cv=self.cv,n_jobs=self.n_jobs), -1)
                self.allGen.add(chromosome)
                if self.verbose:
                    print(chromosome, "score", chromosome.getQuality())
                i += 1

            if self.verbose:
                print("--------------------------------")
                print()
                print("Crossover old gen 2 parents")
                print("--------------------------------------")
            
            # the remains part is the offspring of the 50% best individuals of the old population that mutated in their life
            for i in range(self.pop_size):
                parents = []
                for _ in range(2):
                    equalParents = True
                    while equalParents: # the parents must be different
                        parent_idx = np.random.randint(0, len(mutated_population))
                        equalParents = mutated_population[parent_idx] in parents

                    parents.append(mutated_population[parent_idx])
                # Cross-over and mutation of the son
                isNew = False
                j = 0
                while isNew == False: # Find a completely new chromosome
                    chromosome = BoolChromosome(parents=parents)
                    if j > self.size: # case in which all the possible sons are already in the population we do a mutation to create a new one
                        chromosome.mutate(differentFrom=self.allGen)
                    isNew = chromosome not in self.allGen
                    j += 1

                new_generation[chromosome] = (chromosome.getQuality(X, y, self.estimator, scoring=self.scoring,cv=self.cv,n_jobs=self.n_jobs), -1)
                self.allGen.add(chromosome)
                if self.verbose:
                    print(chromosome, "score", chromosome.getQuality())

            if self.verbose:
                print("--------------------------------")

            # ready for the next generation
            best = self.population[0] # save the best of the old gen
            self.population = new_generation
            gen += 1

        return self
    
    def _get_support_mask(self):
        # return the chromosome with the best score
        return
    
if __name__ == "__main__":
    ## TESTS
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )

    forest = RandomForestClassifier(random_state=0)
    ga = GeneticSelection(forest, pop_size=20)
    ga.fit(X, y)