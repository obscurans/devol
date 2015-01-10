/** Copyright (C) 2015 Jeffrey Tsang. All rights reserved. See /LICENCE.md */

import std.algorithm, std.conv, std.exception, std.functional, std.stdio;
public import std.random;

/* The base class for the gene representation in evolution.
 * Templated (rather circularly) with the final subclass type to force return types to be derived type.
 * Do not inherit directly, specialize to either GeneL or GeneN (see below).
 *
 * Technically alias this could be used, but that is even messier than what is given the abstract methods.
 */
class Gene(G) {
	abstract @property real fitness();

	abstract @property G dup() const
	out(ret) {
		assert(ret !is null, to!string(typeid(G)) ~ ":dup must return non-null copy");
	} body {
		assert(0, to!string(typeid(Gene)) ~ ":dup base class method cannot be used");
	}

	abstract void mutate(ref Random);

	abstract G[2] crossover(const G, ref Random) const
	out(ret) {
		assert(ret[0] !is null && ret[1] !is null, to!string(typeid(G)) ~ ":crossover must return two non-null children");
	} body {
		assert(0, to!string(typeid(Gene)) ~ ":crossover base class method cannot be used");
	}

	override int opCmp(Object other) {
		if (fitness > (cast(Gene)other).fitness) {
			return 1;
		} else if (cachefit == (cast(Gene)other).cachefit) {
			return 0;
		} else {
			return -1;
		}
	}

protected:
	real cachefit; /* Cached fitness value */
}

/* The base class for gene representations that have local fitness functions,
 * that is, fitness can be evaluated independently of other genes */
class GeneL(G) : Gene!G {
	abstract real evaluate();
	abstract override @property G dup() const; /* Duplicated abstract functions for reference */

	abstract override void mutate(ref Random rng)
	out {
		assert(modified, to!string(typeid(G)) ~ ":mutate must set modified flag on exit");
	} body {
		assert(0, to!string(typeid(GeneL)) ~ ":mutate base class method cannot be used");
	}

	abstract override G[2] crossover(const G, ref Random) const;

	final override @property real fitness() {
		if (modified) {
			cachefit = evaluate();
			modified = false;
		}
		return cachefit;
	}

protected:
	bool modified = true; /* Whether the gene was modified after the last fitness evaluation */
}

/* The base class for gene representations that have nonlocal fitness functions,
 * that is, fitness can only be evaluated in groups */
class GeneN(G) : Gene!G {
	abstract override @property G dup() const; /* Duplicated abstract functions for reference */
	abstract override void mutate(ref Random);
	abstract override G[2] crossover(const G, ref Random) const;
	/* Required static method: real[] tournament(G[]) */

	final override @property real fitness() {
		return cachefit;
	}
}

/* Class for running unstructured evolution,
 * that is, in fitness evaluation (if applicable), mating and replacement, every gene can interact with every other */
class UnstructuredEvolution(G)
/* The gene type must inherit either GeneL (local) ... */
if (is(G : GeneL!G) ||
/* or GeneN (nonlocal), but then it must implement real[] G.tournament(G[]) */
(is(G : GeneN!G) && is(typeof(G.tournament([new G])) == real[]))) {
	struct Parameters {
		size_t s_population			= 36;		/* Population size */
		size_t s_elite				= 24;		/* Size of the elite (fitness rank < elite --> automatically copied over per generation) */
		size_t s_eligible			= 24;		/* Size of the eligible parents list (fitness rank >= eligible --> cannot possibly be selected) */
		size_t n_parents			= 12;		/* Number of total parents selected */
		size_t n_children			= 12;		/* Number of total children produced */
		size_t n_replace			= 12;		/* Number of genes in the old population to be replaced:
												 * careful, if childrenReplaceable, then include those in the count */
		bool childrenReplaceable	= false;	/* Whether the newly created children are included in the removable genes at replacement */
		static if (is(G : GeneN!G)) {
			/* For nonlocal-fitness representations, whether to run a full-population tournament at the end of the generation.
			 * Set false for actual tournament-based selection (and implement in the selection function). */
			bool runPosttournament	= true;
		}
		/* Selection operator, input: eligible parents; output: selected parents */
		G[] function(G[], size_t, ref Random) selection		= &DefaultOperators!G.fitnessProportional!(false, true, `a`);
		/* Reproduction operator, input: selected parents; output: children */
		G[] function(G[], size_t, ref Random) reproduction	= &DefaultOperators!G.normalReproduction;
		/* Replacement operator, input: replaceable genes; output: nonreplaced genes - careful, NOT the replaced genes */
		G[] function(G[], size_t, ref Random) replacement	= &DefaultOperators!G.replaceAll;
	}

	Parameters params;	/* Local parameter values */
	alias params this;	/* Causes parameter names to be scoped through the class (so UnstructuredEvolution!G x.s_population is valid) */
	G[] population;		/* Current population */
	uint generation;	/* Current generation number */
	Random rng;			/* Random number generator used for this run */

	/* Generic constructor, auto-initializes the population */
	this(Parameters params = Parameters.init, ref Random rng = rndGen, const G[] seed = []) {
		this.params = params;
		this.rng = rng;
		initPopulation(seed);
	}

	/* (Re)initializes the population, with an optional (not necessarily full) seed population */
	void initPopulation(const G[] seed = []) {
		enforce(seed.length <= s_population, "Too many genes (" ~ to!string(seed.length) ~ ") to include in the initial population (size " ~ to!string(s_population) ~ ")");
		population.length = s_population;
		foreach (i, j; seed) {
			population[i] = j.dup();
		}
		foreach (i; seed.length .. s_population) {
			population[i] = new G(rng);
		}
		static if (is(G : GeneN!G)) {
			if (runPosttournament) {
				G.tournament(population);
			}
		}
		generation = 0;
	}

	/* Run one or more generations */
	void runGeneration(uint count = 1)
	in {
		assert(params.s_population == population.length, "Mismatch between population size (" ~ to!string(params.s_population) ~ ") and actual population (" ~ to!string(population.length) ~ ")");
		assert(params.s_elite <= params.s_population, "Elite size (" ~ to!string(params.s_elite) ~ ") larger than population size (" ~ to!string(params.s_population) ~ ")");
		assert(params.s_eligible <= params.s_population, "Eligible parents list (" ~ to!string(params.s_eligible) ~ ") larger than population size (" ~ to!string(params.s_population) ~ ")");
	} body {
		foreach (i; 0 .. count) {
			G[] temp;

			if (s_eligible < s_population || s_elite > 0) { /* If nontrivial elite or excluded parents, sort population in descending fitness */
				sort!"a>b"(population);
				// TODO: implement random tie-breaking for elitism and parental exclusion
			}

			temp = selection(population[0 .. s_eligible], n_parents, rng); /* Select parents */

			temp = reproduction(temp, n_children, rng); /* Produce children */

			if (childrenReplaceable) {
				/* If children replaceable, new population is elite plus nonreplaced of nonelite and children */
				population = population[0 .. s_elite] ~ replacement(population[s_elite .. $] ~ temp, n_replace, rng);
			} else {
				/* If children not replaceable, new population is elite plus children plus nonreplaced nonelite */
				population = population[0 .. s_elite] ~ temp ~ replacement(population[s_elite .. $], n_replace, rng);
			}
			s_population = population.length; /* In case reproduction and replacement are unbalanced */

			static if (is(G : GeneN!G)) {
				if (runPosttournament) {
					G.tournament(population);
				}
			}
			generation++;
		}
	}
}

/* Pre-written default operators for selection/reproduction/replacement */
template DefaultOperators(G) {
	/* Fitness-proportional selection,
	 * with compile-time parameters to do reproduction/replacement, whether repetitions are allowed,
	 * and an optional function to weight the fitness values.
	 * This allows, for example, fitness-inversely proportional selection (use `map!"1/a"(a).array`). */
	G[] fitnessProportional(bool replace, bool repeat, alias weighting = (real[] x) { return x; })
	(G[] input, size_t count, ref Random rng)
	/* in replacement, repetitions disallowed;
	 * weighting allowed to be a unaryFun string as well, type function real[](real[]) */
	if (!(replace && repeat) && is(typeof(unaryFun!weighting([.0L])) == real[]))
	in {
		static if (!repeat) {
			static if (!replace) {
				assert(input.length >= count, "For fitnessProportional(selection), without repetitions, at least as many candidate parents (only " ~ to!string(input.length) ~ ") as required parents (" ~ to!string(count) ~ ") needed");
			} else {
				assert(input.length >= count, "For fitnessProportional(replacement), at least as many removable genes (only " ~ to!string(input.length) ~ ") as required removals (" ~ to!string(count) ~ ") needed");
			}
		}
	} out(ret) {
		static if (!replace) {
			assert(ret.length == count);
		} else {
//			assert(ret.length == parameters.replace);
		}
	} body {
		real total = 0, dart;
		real[] weight; /* Stores the fitness value weightings */
		G[] ret;

		void remove(size_t i) { /* Helper function to remove the ith gene from consideration */
			if (i < input.length - 1) {
				input = input[0 .. i] ~ input[i + 1 .. $];
				weight = weight[0 .. i] ~ weight[i + 1 .. $];
			} else {
				input = input[0 .. i];
				weight = weight[0 .. i];
			}
		}

		weight.length = input.length; /* Collect and weight all fitness values */
		foreach (i, j; input) {
			weight[i] = j.fitness;
		}
		weight = unaryFun!weighting(weight);
		for (size_t i = 0; i < input.length;) {
			if (weight[i] <= 0) { /* Preemptively remove nonpositive-fitness-weighted genes from candidate parents */
				remove(i);
			} else {
				total += weight[i]; /* Accumulate total weights as normalizing factor */
				i++;
			}
		}

		static if (!replace) { /* Initialize correct number of genes to select */
			ret.length = count;
		}
		foreach (i; 0 .. count) {
			if (input.length == 1) { /* A single candidate gene is automatically selected */
				ret[i] = input[0];
				continue;
			}
			dart = uniform(0, total, rng);
			foreach (j; 0 .. input.length) {
				if ((dart -= weight[j]) <= 0) { /* Linear search through the list, this could be improved */
					static if (!replace) {
						ret[i] = input[j];
					}
					static if (!repeat) { /* If without replacement, remove gene from further consideration and reweight */
						total -= weight[j];
						remove(j);
					}
					break;
				}
			}
		}

		static if (!replace) {
			return ret;
		} else {
			return input; /* If replacement, invert selection by returning non-selected genes */
		}
	}

	/*template fitnessProportionalRepetition(alias weighting = (real[] x) { return x; }) { //TODO: currently fails with default instantiation (lambda cannot be used to chain templates?)
		alias fitnessProportional!(false, true, weighting) fitnessProportionalRepetition;
	}*/

	/* Vanilla reproduction: each pair in order in the selected parents list undergo crossover, then both children are mutated */
	G[] normalReproduction(G[] input, size_t count, ref Random rng)
	in {
		assert(input.length == count, "For normalReproduction, number of selected parents (" ~ to!string(input.length) ~ ") must equal number of children produced (" ~ to!string(count) ~ ")");
		assert(count % 2 == 0, "For normalReproduction, number of children produced (" ~ to!string(count) ~ ") must be even");
	} body {
		G[] ret;
		ret.length = count;
		foreach (i; 0 .. count / 2) {
			ret[i * 2 .. i * 2 + 2] = input[i * 2].crossover(input[i * 2 + 1], rng);
		}
		foreach (i; ret) {
			i.mutate(rng);
		}
		return ret;
	}

	/* Complete replacement, used mostly in elitist algorithms */
	G[] replaceAll(G[] input, size_t count, ref Random rng)
	in {
		assert(input.length == count, "For replaceAll, number of removable genes (" ~ to!string(input.length) ~ ") must equal required removals (" ~ to!string(count) ~ ")");
	} body {
		return []; /* Nothing is left */
	}
}

