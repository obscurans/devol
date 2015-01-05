/** Copyright (C) 2015 Jeffrey Tsang. All rights reserved. See /LICENCE.md */

import std.algorithm, std.conv, std.math, std.stdio;
import devol;
/*extern (C) { //currently LAPACK not used
	void dgesv_(const int *n, const int *nrhs, const double *a, const int *lda, int *ipiv, double *b, const int *ldb, int *info);
}*/

class DFT : GeneN!DFT {
	enum Move {
		C,	/* Cooperate */
		D	/* Defect */
	}
	static immutable Move[] Moves = [ Move.C, Move.D ]; //TODO: find better way to iterate through enum
	static immutable size_t movec = Moves.length;

	static {								/* Global (DFT-wide) parameters */
		size_t new_state_count		= 80;	/* Number of states for randomly initialized automata (all automata have this state count) */
		uint mut_count				= 1;	/* Number of repeated point mutations per mutation operation */
		real mut_init_state			= 0.05;	/* Probability of mutating initial state */
		bool mut_init_state_null	= true;	/* Whether mutating initial state can be null (do nothing) */
		real mut_init_act			= 0.05;	/* Probability of mutating initial action */
		bool mut_init_act_null		= true;	/* Whether mutating initial action can be null */
		real mut_trans				= 0.4;	/* Probability of mutating a random transition */
		bool mut_trans_null			= true;	/* Whether mutating transition can be null */
		real mut_act				= 0.5;	/* Probability of mutating a random response action */
		bool mut_act_null			= true;	/* Whether mutating response can be null */
		uint cross_count			= 1;	/* Number of independent 2-point crossovers per crossover operation */
		bool cross_null				= true;	/* Whether crossover can be null */
		real[movec][movec] score	= [ [ 3 /* C/opp.C -> R */, 0 /* C/opp.D -> S */ ],
									    [ 5 /* D/opp.C -> T */, 1 /* D/opp.D -> P */ ] ]; /* Payoff matrix */
		uint eval_rounds			= 150;	/* Number of rounds to simulate for fitness evaluation, set to zero or negative for infinite horizon */
		real eval_alpha				= 1;	/* Geometric weighting parameter for fitness evaluation, set to one for unweighted */
	}

	struct State {
		size_t[movec] trans;	/* Transition state array, indexed by input move */
		Move[movec] act;		/* Response action array, indexed by input move */

		/* Restore field-based constructor */
		this(size_t[movec] trans, Move[movec] act) {
			this.trans = trans;
			this.act = act;
		}

		/* Random initialization constructor, default state count */
		this(ref Random rng) {
			this(DFT.new_state_count, rng);
		}

		/* Random initialization constructor */
		this(size_t length, ref Random rng) {
			foreach (m; Moves) {
				trans[m] = uniform(0, length, rng);
				act[m] = uniform!Move(rng);
			}
		}

		string toString() const {
			string ret = "";
			foreach (m; Moves) {
				ret ~= /*to!string(m) ~ ":" ~*/ to!string(trans[m]) ~ "/" ~ to!string(act[m]);
				if (m < movec - 1) {
					ret ~= " ";
				}
			}
			return ret;
		}
	}

	size_t init_state;				/* Initial state */
	Move init_act;					/* Initial action */
	State[] states;					/* Array of automaton states */
	private State[] st_canonical;	/* Canonicalized automaton states: initial state is 0, all reachable states renumbered in breadth-first order */
	private State[] st_minimal;		/* Minimized automaton states: initial state is 0, minimal-state count equivalent canonicalized automaton */
	private size_t cur_state = -1;	/* Current state, used for direct game simulations */
	private bool modified_c = true;	/* Whether modified since last state canonicalization */
	private bool modified_m = true;	/* Whether modified since last state minimization */

	/* Restore default constructor */
	this() {}

	/* Copy constructor */
	this(const DFT other) {
		init_state = other.init_state;
		init_act = other.init_act;
		states = other.states.dup;
		cur_state = other.cur_state;
	}

	/* Random initialization constructor */
	this(ref Random rng) {
		init_state = uniform(0, new_state_count, rng);
		init_act = uniform!Move(rng);
		states.length = new_state_count;
		foreach (ref state; states) {
			state = State(new_state_count, rng);
		}
	}

	override string toString() const {
		return to!string(cachefit) ~ ":" ~ to!string(init_state) ~ "/" ~ to!string(init_act) ~ "-" ~ to!string(states);
	}

	override @property DFT dup() const {
		return new DFT(this);
	}

	/* One-point mutation, with repeats and probabilites controlled through parameters */
	override void mutate(ref Random rng) {
		cur_state = -1; /* Unset current state pointer */
		modified_c = modified_m = true; /* Set modified flags */
		foreach (i; 0 .. mut_count) {
			real dart = uniform(0, 1., rng);

			dart -= mut_init_state;
			if (dart < 0) {
				size_t old = init_state;
				do {
					init_state = uniform(0, states.length, rng);
				} while (!mut_init_state_null && old == init_state); /* Repeat if initial state unchanged and null mutation not allowed */
				continue;
			}

			dart -= mut_init_act;
			if (dart < 0) {
				Move old = init_act;
				do {
					init_act = uniform!Move(rng);
				} while (!mut_init_act_null && old == init_act);
				continue;
			}

			dart -= mut_trans;
			if (dart < 0) {
				size_t* target = &states[uniform(0, states.length, rng)].trans[uniform!Move(rng)]; /* Select random transition from random state */
				size_t old = *target;
				do {
					*target = uniform(0, states.length, rng);
				} while (!mut_trans_null && old == *target);
				continue;
			}

			dart -= mut_act;
			if (dart < 0) { /* Still tested since if mutation probabilities sum < 1, forced null mutations possible */
				Move* target = &states[uniform(0, states.length, rng)].act[uniform!Move(rng)]; /* Select random action from random state */
				Move old = *target;
				do {
					*target = uniform!Move(rng);
				} while (!mut_act_null && old == *target);
			}
		}
	}

	/* Two-point crossover, keeping entire states as atomic objects.
	 * The initial "state" is considered a separate atomic object after all actual states.
	 * Repeats and possible null controlled through parameters. */
	override DFT[2] crossover(const DFT other, ref Random rng) const 
	in {
		assert(this.states.length == other.states.length, "Crossover impossible when state counts of parent automata (" ~ to!string(this.states.length) ~ ", " ~ to!string(other.states.length) ~ ") do not match");
	} body {
		DFT[2] ret;
		size_t dart1, dart2;

		ret[0] = this.dup; /* Copy parent automata directly */
		ret[0].cur_state = -1; /* But unset current state pointer and set modified flags */
		ret[0].modified_c = ret[0].modified_m = true;
		ret[1] = other.dup;
		ret[1].cur_state = -1;
		ret[1].modified_c = ret[1].modified_m = true;

		foreach (i; 0 .. cross_count) {
			dart1 = uniform(0, states.length + 1, rng);
			do {
				dart2 = uniform(0, states.length + 1, rng);
			} while (!cross_null && dart1 == dart2); /* Force different points if no null crossovers */

			if (dart1 < dart2) { /* States in range [dart1, dart2) are swapped */
				swapRanges(ret[0].states[dart1 .. dart2], ret[1].states[dart1 .. dart2]);
			} else if (dart1 > dart2) { /* States in range [0 .. dart2) U [dart1 .. end] U initial are swapped */
				swapRanges(ret[0].states[0 .. dart2], ret[1].states[0 .. dart2]);
				swapRanges(ret[0].states[dart1 .. $], ret[1].states[dart1 .. $]);
				swap(ret[0].init_state, ret[1].init_state);
				swap(ret[0].init_act, ret[1].init_act);
			} /* Null crossover if crossover points match */
		}

		return ret;
	}

	/* Round-robin tournament on the input (ignores this). May eventually be split statically. */
	override real[] tournament(DFT[] input) {
		real[] ret;
		real[movec][movec] match;
		real[movec][movec][] results;
		ret.length = input.length;
		results.length = input.length;
		foreach (ref r1; results) { /* Initialize results array */
			foreach (ref r2; r1) {
				r2[] = 0;
			}
		}

		foreach (i; 0 .. ret.length - 1) {
			foreach (j; i + 1 .. ret.length) { /* Symmetric game, so run upper triangle for round-robin */
				if (eval_alpha == 1) { /* If evaluation is unweighted */
					if (eval_rounds > 0 && eval_rounds < 0xFFFF) { /* If evaluation rounds is set and not insane */
						match = input[i].simulate(input[j], eval_rounds); /* Dispatch the correct simulation function, could be made static */
					} else {
						match = input[i].simulate_infinite(input[j]);
					}
				} else {
					if (eval_rounds > 0 && eval_rounds < 0xFFFF) {
						match = input[i].simulate_weighted(input[j], eval_alpha, eval_rounds);
					} else {
						match = input[i].simulate_infinite_weighted(input[j], eval_alpha);
					}
				}
				foreach (m1; Moves) {
					foreach (m2; Moves) {
						results[i][m1][m2] += match[m1][m2]; /* Accumulate counts from this match for the automata */
						results[j][m2][m1] += match[m1][m2]; /* Counts for opponent have order of the moves reversed for result-type */
					}
				}
			}
		}

		ret[] = 0;
		foreach (i; 0 .. ret.length) {
			foreach (m1; Moves) {
				foreach (m2; Moves) {
					ret[i] += results[i][m1][m2] * score[m1][m2]; /* Total score is dot product of result-type count and score matrix */
				}
			}
		}
		ret[] /= ret.length - 1; /* Normalize to per-round scores */

		foreach (i; 0 .. ret.length) {
			input[i].cachefit = ret[i]; /* Store fitness results for participating automata */
		}
		return ret;
	}

	/* Return read-only view of the canonicalized automaton */
	@property const(State[]) states_canonical() {
		if (modified_c) {
			canonicalizeStates();
		}
		return st_canonical;
	}

	/* Return read-only view of the minimized automaton */
	@property const(State[]) states_minimal() {
		if (modified_m) {
			minimizeStates();
		}
		return st_minimal;
	}

private:
	/* Make the initial move and transition */
	Move play_init() {
		cur_state = init_state;
		return init_act;
	}

	/* Make the next move and transition */
	Move play_next(Move input) {
		size_t cur = cur_state;
		cur_state = states[cur].trans[input];
		return states[cur].act[input];
	}

	/* Make the initial move and transition, using minimized representation */
	Move play_mininit()
	in {
		assert(!modified_m, "Minimized automaton out of sync");
	} body {
		cur_state = 0;
		return init_act;
	}

	/* Make the next move and transition, using minimized representation */
	Move play_minnext(Move input)
	in {
		assert(!modified_m, "Minimized automaton out of sync");
	} body {
		size_t cur = cur_state;
		cur_state = st_minimal[cur].trans[input];
		return st_minimal[cur].act[input];
	}

	/* Direct simulation of the game, up to a specified number of rounds. Returns normalized counts of move-pairs (results).
	 * Uses real instead of int for compatibility with everything else. */
	real[movec][movec] simulate(DFT other, uint rounds)
	out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-10, 1e-10), "simulate returned " ~ to!string(total) ~ " when 1 expected"); //TODO: find sum(real[][])
	} body {
		real[movec][movec] ret = 0;
		Move sm, om, sm2, om2;

		sm = this.play_init();
		om = other.play_init();
		ret[sm][om]++;

		foreach (i; 1 .. rounds) {
			sm2 = this.play_next(om);
			om2 = other.play_next(sm);
			ret[sm = sm2][om = om2]++;
		}

		foreach (m; Moves) {
			ret[m][] /= rounds; /* Normalize total counts */
		}
		return ret;
	}

	/* Direct truncated-geometric(alpha)-weighted simulation of the game, up to a specified number of rounds.
	 * Returns distribution over move-pairs (results) */
	real[movec][movec] simulate_weighted(DFT other, real alpha, uint rounds)
	in {
		assert(alpha >= 0 && alpha < 1, "For simulate_weighted, alpha parameter (" ~ to!string(alpha) ~ ") must be in [0, 1)");
	} out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-10, 1e-10)); //TODO: find sum(real[][])
	} body {
		const real factor = (1 - alpha) / (1 - alpha ^^ rounds); /* Total weighting is sum(n=0..rounds-1)alpha^n = (1-alpha^n)/(1-alpha) */
		real[movec][movec] ret = 0;
		real alphan = 1; /* Stores alpha^n by successive multiplication */
		Move sm, om, sm2, om2;

		sm = this.play_init();
		om = other.play_init();
		ret[sm][om]++;

		foreach (i; 1 .. rounds) {
			sm2 = this.play_next(om);
			om2 = other.play_next(sm);
			alphan *= alpha;
			ret[sm = sm2][om = om2] += alphan; /* Geometric weighting by alpha^n */
		}

		foreach (m; Moves) {
			ret[m][] *= factor; /* Normalize total counts */
		}
		return ret;
	}

	/* Simulates the game and returns the infinite-horizon per-round averaged distribution over move-pairs (results).
	 *
	 * Using the property that this is a deterministic finite Markov chain, the formulation can be massively simplified.
	 * Run the cross-product chain until it reaches a cycle, then average over the cycle only.
	 * This unfortunately relies on the fact that 1^n=1 and is completely inapplicable to nondeterministic systems.
	 */
	real[movec][movec] simulate_infinite(bool minimize = false)(DFT other)
	out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-10, 1e-10)); //TODO: find sum(real[][])
	} body {
		struct Count { /* To force the associative array to behave properly */
			uint[movec][movec] c;
		}

		real[movec][movec] ret = 0;
		Count[size_t] count;	/* (3)+2-dimensional array: Markov state = (state of 1 x state of 2 x last move of 1 x last move of 2)
								 * -> aggregate counts of move-pairs of the simulation up to this Markov state.
								 * Associative array used as this is really sparse.
								 * As a side bonus, inExpression is a test for past visit */
		size_t mid, mid2;		/* Indices into the Markov chain */
		Move sm, om, sm2, om2;

		static if (minimize) {
			states_minimal; /* Ensure both automata have been state-minimized */
			other.states_minimal;

			const size_t n1_states = st_minimal.length;
			const size_t n2_states = other.st_minimal.length;
		} else {
			const size_t n1_states = states.length;
			const size_t n2_states = other.states.length;
		}
		const size_t m_states = n1_states * n2_states * movec * movec; /* Implementation choice: Markov chain index linearized to avoid double indirection */

		size_t index(size_t q1, size_t q2, Move m1, Move m2) { /* Indexing function for state x state x move x move */
			return ((q1 * n2_states + q2) * movec + m1) * movec + m2;
		}

		static if (minimize) {
			sm = play_mininit();
			om = other.play_mininit();
		} else {
			sm = play_init();
			om = other.play_init();
		}
		mid = index(cur_state, other.cur_state, sm, om);
		count[mid] = Count();
		count[mid].c[sm][om] = 1; /* At this Markov state, the only move-pair ever played is sm-om */

		while (1) {
			static if (minimize) {
				sm2 = play_minnext(om);
				om2 = play_minnext(sm);
			} else {
				sm2 = play_next(om);
				om2 = play_next(sm);
			}
			mid2 = index(cur_state, other.cur_state, sm2, om2);
			if (mid2 in count) { /* The Markov chain has closed the cycle */
				break;
			}

			count[mid2] = count[mid]; /* Copy over the aggregate counts */
			count[mid2].c[sm2][om2]++; /* And increment the currently played pair */

			sm = sm2;
			om = om2;
			mid = mid2;
		}

		/* At this point mid is the last state in the cycle, which closes to mid2 */
		foreach (m1; Moves) {
			foreach (m2; Moves) {
				ret[m1][m2] = count[mid].c[m1][m2] - count[mid2].c[m1][m2]; /* This subtraction counts the aggregate moves within the cycle only */
			}
		}
		ret[sm2][om2]++; /* Add back the cycle-closing move pair */

		real total = 0;
		foreach (r1; ret) {
			foreach (r2; r1) {
				total += r2;
			}
		}
		foreach (ref r1; ret) {
			r1[] /= total; /* Normalize total counts */
		}
		return ret;
	}

	/* Simulates the game and returns the infinite-horizon geometric(alpha)-weighted distribution over move-pairs (results).
	 *
	 * As with simulate_infinite above, heavily using the property that this is a deterministic finite Markov chain.
	 * Run the cross-product chain until it closes a cycle, then calculate the tail weighting over the cycle to complete.
	 * Method completely inapplicable to nondeterministic systems.
	 */
	real[movec][movec] simulate_infinite_weighted(DFT other, real alpha)
	in {
		assert(alpha >= 0 && alpha < 1, "For simulate_infinite_weighted, alpha parameter (" ~ to!string(alpha) ~ ") must be in [0, 1)");
	} out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-10, 1e-10)); //TODO: find sum(real[][])
	} body {
		real[movec][movec] ret = 0;

		states_minimal; /* Ensure both automata have been state-minimized */
		other.states_minimal;

		//TODO

		return ret;
	}

	/* Canonicalize the state representation, by renumbering all the states so that initial is 0,
	 * and other reachable states are numbered in breadth-first order.
	 *
	 * Maintains a mapping from representation state-number to canonicalized state-number,
	 * then feeds all states through a queue to traverse all reachable states.
	 */
	void canonicalizeStates()
	out {
		assert(st_canonical.length > 0 && st_canonical.length <= states.length);
	} body {
		size_t[] mapping; /* Mapping of representation state-number to the canonicalized state-number */

		mapping.length = states.length;
		mapping[] = -1;
		st_canonical.length = 1; /* Use the output automaton state list as its own queue of visited states */
		mapping[init_state] = 0;
		st_canonical[0].trans[0] = init_state; /* Start the queue with the initial state;
												* as a hack, the old-state-number is stored here before processing */

		for (size_t ct = 0; ct < st_canonical.length; ct++) { /* Loop until the queue is exhausted */
			size_t oldst = st_canonical[ct].trans[0]; /* Retrieve old-state-number */
			foreach (m; Moves) {
				size_t oldtrans = states[oldst].trans[m]; /* Take the transition target from the old state */
				if (mapping[oldtrans] == -1) { /* If this target state has yet to be mapped */
					mapping[oldtrans] = st_canonical.length;
					st_canonical[ct].trans[m] = st_canonical.length; /* Create new state for it and append to the end of the processing queue */
					st_canonical.length++;
					st_canonical[$ - 1].trans[0] = oldtrans;
				} else {
					st_canonical[ct].trans[m] = mapping[oldtrans]; /* Otherwise set the transition as mapped */
				}
				st_canonical[ct].act[m] = states[oldst].act[m]; /* Copy the action over */
			}
		}

		modified_c = false;
	}

	/* Table-marking algorithm for minimizing DFA, extended to DFT simply by constructing the initial partition using single-character output,
	 * i.e. equivalence classes under identical output on all input letters.
	 *
	 * General idea: if states are indistinguishable under any input, they can be combined without affecting the automaton.
	 * Clearly, accepting states are distinguishable from nonaccepting states (DFT: read "if they output differently under any input")
	 * Inductively, if delta(q1,m) and delta(q2,m) are distinguishable, then q1 and q2 are distinguishable
	 * The basic algorithm runs a fixed-point iteration algorithm on the inductive definition, then merges states proven indistinguishable
	 *
	 * Algorithm improved to one-pass by maintaining a dependency list for back-marking:
	 * if q1'=delta(q1,m) and q2'=delta(q2,m) is unmarked, then (q1,q2) depends on (q1',q2') to remain unmarked.
	 * Marking (q1',q2') recursively marks all (q1,q2) that depend on it.
	 * Complexity is still O(n^2*s), proved easily by considering there are only nC2*s possible dependency marks in existence,
	 * and recursive_mark does work proportional to number of dependency marks followed.
	 *
	 * TODO: figure out if Hopcroft's algorithm works here. */
	void minimizeStates()
	out {
		assert(st_minimal.length > 0 && st_minimal.length <= st_canonical.length && st_minimal.length <= states.length);
	} body {
		bool[] marked; /* Implementation choice: square arrays for state x state linearized here, to avoid double indirection.  */
		size_t[][] backmark; /* backmark[i,j] = array of state pair indices depending on (i,j) */
		size_t[] mapping; /* Eventually used to canonicalize the minimized automaton */

		if (modified_c) { /* State minimization operates on canonicalized copy */
			canonicalizeStates();
		}

		const size_t n_states = st_canonical.length;
		marked.length = n_states * n_states;
		backmark.length = n_states * n_states;
		mapping.length = n_states;

		/* Indexing function for state x state.
		 * Possible improvement: use triangular packed indexing, index of (i,j), i<j, = (n-1)C2 - (n-i)C2 + j-i-1. */
		size_t index(size_t i, size_t j) {
			return i * n_states + j;
		}

		void recursiveMark(size_t ij) { /* Recursive marking of the distinguishable table */
			marked[ij] = true;
			foreach (ij2; backmark[ij]) {
				if (!marked[ij2]) {
					recursiveMark(ij2);
				}
			}
		}

		foreach (i; 0 .. n_states) {
			size_t id = 0; /* Unique ID indexing all output-combinations for input alphabet */
			foreach (m; Moves) {
				id = id * movec + st_canonical[i].act[m];
			}
			mapping[i] = id; /* Use the mapping table for now */
		}
		foreach (i; 0 .. n_states - 1) {
			foreach (j; i + 1 .. n_states) {
				marked[index(i,j)] = mapping[i] != mapping[j]; /* Initialize distinguishability table */
			}
		}

		foreach (i; 0 .. n_states - 1) {
			foreach (j; i + 1 .. n_states) {
				size_t ij = index(i,j); /* Cache index for reuse */
				if (!marked[ij]) {
					foreach (m; Moves) {
						size_t i2 = st_canonical[i].trans[m]; /* Find the transition-states from both i and j, same input */
						size_t j2 = st_canonical[j].trans[m];
						if (i2 != j2) { /* If exactly equal states, that cannot be a distinguisher */
							if (i2 > j2) {
								swap(i2, j2);
							}
							size_t ij2 = index(i2,j2);
							if (marked[ij2]) {
								recursiveMark(ij);
								break; /* No need to check other moves */
							} else {
								backmark[ij2] ~= ij; /* Directly store index (to reduce computation) into dependency list */
							}
						}
					}
				}
			}
		}
		backmark.length = 0; /* Allow GC to collect this if needed */

		/* Use the same algorithm as canonicalizing, except when mapping a new state, map all states indistinguishable to it as well */
		void mapIndistinguishable(size_t i)
		in {
			assert(mapping[i] == -1);
		} body {
			mapping[i] = st_minimal.length;
			foreach (j; 0 .. i) {
				if (!marked[index(j,i)]) {
					assert(mapping[j] == -1);
					mapping[j] = st_minimal.length;
				}
			}
			foreach (j; i + 1 .. n_states) {
				if (!marked[index(i,j)]) {
					assert(mapping[j] == -1);
					mapping[j] = st_minimal.length;
				}
			}
			st_minimal.length++;
			st_minimal[$ - 1].trans[0] = i;
		}

		mapping[] = -1;
		st_minimal.length = 0; /* mapIndistinguishable will handle queue initialization */
		mapIndistinguishable(0); /* Initial state in the canonical automaton (both input and output) is number 0 */

		for (size_t ct = 0; ct < st_minimal.length; ct++) {
			size_t oldst = st_minimal[ct].trans[0]; /* Retrieve old-state-number */
			foreach (m; Moves) {
				size_t oldtrans = st_canonical[oldst].trans[m]; /* Take the transition target from the old state */
				if (mapping[oldtrans] == -1) { /* If this target state has yet to be mapped, do it */
					mapIndistinguishable(oldtrans);
				}
				st_minimal[ct].trans[m] = mapping[oldtrans]; /* Set the transition as mapped and copy the action */
				st_minimal[ct].act[m] = st_canonical[oldst].act[m];
			}
		}

		modified_m = false;
	}
}

