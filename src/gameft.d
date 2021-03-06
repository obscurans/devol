/** Copyright (C) 2015 Jeffrey Tsang. All rights reserved. See /LICENCE.md */

import std.algorithm, std.conv, std.math, std.stdio;
import devol;
extern (C) { //currently LAPACK not used
	void dgesv_(const int *n, const int *nrhs, const double *a, const int *lda, int *ipiv, double *b, const int *ldb, int *info);
}

class DFT {
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

	@property DFT dup() const {
		return new DFT(this);
	}

	/* One-point mutation, with repeats and probabilites controlled through parameters */
	void mutate(ref Random rng) {
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
	DFT[2] crossover(const DFT other, ref Random rng) const
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

	/* Computes the 8-channel fingerprint (CC,CD,DC,DD for opponent cooperates and defects) at a single point (y,z) with given alpha */
	real[8] fp_point(real y, real z, real alpha)
	in {
		assert(movec == 2, "Fingerprinting only implemented for movec = 2 (current: " ~ to!string(movec) ~ ")");
		assert(alpha >= 0 && alpha <= 1, "Invalid alpha parameter: " ~ to!string(alpha));
	} body {
		if (modified_m) { /* Fingerprinting uses minimized representation */
			minimizeStates();
		}

		int info;
		int[] ipiv;
		const size_t nstates = st_minimal.length;
		const int n4 = cast(int)nstates * 4;
		immutable int nrhs = 2;
		ipiv.length = n4;

		double[] q;
		q.length = n4 * 2;
		q[] = 0;
		q[init_act * 2] = 1 - alpha;
		q[init_act * 2 + n4 + 1] = 1 - alpha;

		double[] a;
		a.length = n4 * n4;
		a[] = 0;
		foreach (i; 0 .. nstates) {
			a[n4 * i * 4 + (st_minimal[i].trans[0] * 2 + st_minimal[i].act[0]) * 2] = -y * alpha;
			a[n4 * i * 4 + (st_minimal[i].trans[0] * 2 + st_minimal[i].act[0]) * 2 + 1] = (y - 1) * alpha;
			a[n4 * (i * 4 + 1) + (st_minimal[i].trans[1] * 2 + st_minimal[i].act[1]) * 2] = -y * alpha;
			a[n4 * (i * 4 + 1) + (st_minimal[i].trans[1] * 2 + st_minimal[i].act[1]) * 2 + 1] = (y - 1) * alpha;
			a[n4 * (i * 4 + 2) + (st_minimal[i].trans[0] * 2 + st_minimal[i].act[0]) * 2] = -z * alpha;
			a[n4 * (i * 4 + 2) + (st_minimal[i].trans[0] * 2 + st_minimal[i].act[0]) * 2 + 1] = (z - 1) * alpha;
			a[n4 * (i * 4 + 3) + (st_minimal[i].trans[1] * 2 + st_minimal[i].act[1]) * 2] = -z * alpha;
			a[n4 * (i * 4 + 3) + (st_minimal[i].trans[1] * 2 + st_minimal[i].act[1]) * 2 + 1] = (z - 1) * alpha;
		}
		foreach (i; 0 .. n4) {
			a[i * (n4 + 1)] += 1;
		}

		dgesv_(&n4, &nrhs, a.ptr, &n4, ipiv.ptr, q.ptr, &n4, &info);

		real[8] ret = 0;
		foreach (i; 0 .. 4) {
			foreach (j; 0 .. nstates) {
				ret[i] += q[i + j * 4];
			}
			foreach (j; nstates .. 2 * nstates) {
				ret[i + 4] += q[i + j * 4];
			}
		}
		return ret;
	}

	/* Computes the colour components of the fingerprint structure predictor (ALLD,ALLC,TFT,PSY,INIT) with the given alpha */
	real[5] fp_component(real alpha)
	in {
		assert(movec == 2, "Fingerprinting only implemented for movec = 2 (current: " ~ to!string(movec) ~ ")");
		assert(alpha >= 0 && alpha <= 1, "Invalid alpha parameter: " ~ to!string(alpha));
	} body {
		if (modified_m) { /* Fingerprinting uses minimized representation */
			minimizeStates();
		}

		int info;
		int[] ipiv;
		const int nstates = cast(int)st_minimal.length;
		immutable int nrhs = 1;
		ipiv.length = nstates;

		double[] q;
		q.length = nstates;
		q[] = 0;
		q[0] = 1 - alpha;

		double[] a;
		a.length = nstates * nstates;
		a[] = 0;
		foreach (i; 0 .. nstates) {
			a[nstates * i + st_minimal[i].trans[0]] = -0.5 * alpha;
			a[nstates * i + st_minimal[i].trans[1]] -= 0.5 * alpha;
		}
		foreach (i; 0 .. nstates) {
			a[i * (nstates + 1)] += 1;
		}

		dgesv_(&nstates, &nrhs, a.ptr, &nstates, ipiv.ptr, q.ptr, &nstates, &info);

		real[5] ret = 0;
		foreach (i; 0 .. nstates) {
			if (st_minimal[i].act[0]) {
				if (st_minimal[i].act[1]) {
					ret[0] += q[i];
				} else {
					ret[3] += q[i];
				}
			} else {
				if (st_minimal[i].act[1]) {
					ret[2] += q[i];
				} else {
					ret[1] += q[i];
				}
			}
		}
		ret[] *= alpha;
		if (init_act) {
			ret[4] = alpha - 1;
		} else {
			ret[4] = 1 - alpha;
		}

		return ret;
	}

private:
	/* Make the initial move and transition, with compile-time switch to use minimized representation */
	Move play_init(bool minimize = false)()
	in {
		static if (minimize) {
			assert(!modified_m, "Minimized automaton out of sync");
		}
	} body {
		static if (minimize) {
			cur_state = 0;
			return init_act;
		} else {
			cur_state = init_state;
			return init_act;
		}
	}

	/* Make the next move and transition, with compile-time switch to use minimized representation */
	Move play_next(bool minimize = false)(Move input)
	in {
		static if (minimize) {
			assert(!modified_m, "Minimized automaton out of sync");
		}
	} body {
		size_t cur = cur_state;
		static if (minimize) {
			cur_state = st_minimal[cur].trans[input];
			return st_minimal[cur].act[input];
		} else {
			cur_state = states[cur].trans[input];
			return states[cur].act[input];
		}
	}

	/* Direct simulation of the game, up to a specified number of rounds. Returns distribution over move-pairs (results).
	 * Uses real instead of int for compatibility with everything else.
	 *
	 * Compile-time overload (if weighted): direct truncated-geometric(alpha)-weighted simulation of the game */
	real[movec][movec] simulate(bool weighted = false)(DFT other, uint rounds, real alpha = 1)
	in {
		static if (weighted) {
			assert(alpha >= 0 && alpha < 1, "For simulate!weighted, alpha parameter (" ~ to!string(alpha) ~ ") must be in [0, 1)");
		}
	} out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-15, 1e-15), "simulate returned " ~ to!string(total) ~ " when 1 expected"); //TODO: find sum(real[][])
	} body {
		static if (weighted) {
			const real factor = (1 - alpha) / (1 - alpha ^^ rounds); /* Total weighting is sum(n=0..rounds-1)alpha^n = (1-alpha^n)/(1-alpha) */
			real alphan = 1; /* Stores alpha^n by successive multiplication */
		}
		real[movec][movec] ret = 0;
		Move sm, om, sm2, om2;

		sm = this.play_init();
		om = other.play_init();
		ret[sm][om]++;

		foreach (i; 1 .. rounds) {
			sm2 = this.play_next(om);
			om2 = other.play_next(sm);
			static if (weighted) {
				alphan *= alpha;
				ret[sm = sm2][om = om2] += alphan; /* Geometric weighting by alpha^n */
			} else {
				ret[sm = sm2][om = om2]++;
			}
		}

		foreach (m; Moves) {
			static if (weighted) {
				ret[m][] *= factor; /* Normalize total weighted counts */
			} else {
				ret[m][] /= rounds; /* Normalize total counts */
			}
		}
		return ret;
	}

	/* Simulates the game and returns the (possibly infinite-horizon) per-round averaged distribution over move-pairs (results).
	 *
	 * Using the property that this is a deterministic finite Markov chain, the formulation can be massively simplified.
	 * Run the cross-product chain until it reaches a cycle, then average over the cycle only.
	 * This unfortunately relies on the fact that 1^n=1 and is completely inapplicable to nondeterministic systems.
	 *
	 * Compile-time overload (if weighted): returns the geometric(alpha)-weighted infinite-horizon distribution over move-pairs (results).
 	 * Run the cross-product chain until it closes a cycle, then calculate the tail weighting over the cycle to complete.
	 *
	 * Can also be used for non-infinite horizon simulations, by weighting the loop correctly as many times to fill the number of rounds,
	 * then adding back the possibly incomplete-cycle tail. While this gives effectively constant expected time (for the same machines)
	 * regardless of horizon, there is extremely significant overhead for bookkeeping and this is not faster until several hundred rounds.
	 * There is a parameter (eval_rounds_switchover) controlling when a finite-horizon simulation starts using this function.
	 */
	real[movec][movec] simulate_loop(bool weighted = false, bool infinite = false, bool minimize = false)
	(DFT other, uint rounds = 0, real alpha = 1)
	in {
		static if (weighted) {
			assert(alpha >= 0 && alpha < 1, "For simulate_loop!weighted, alpha parameter (" ~ to!string(alpha) ~ ") must be in [0, 1)");
		}
		static if (!infinite) {
			assert(rounds > 0, "For simulate_loop!(!infinite), rounds parameter (0) must be greater than 0");
		}
	} out(ret) {
		real total = 0;
		foreach (d1; ret) {
			foreach (d2; d1) {
				total += d2;
			}
		}
		assert(approxEqual(total, 1, 1e-15, 1e-15), "simulate_loop returned " ~ to!string(total) ~ " when 1 expected"); //TODO: find sum(real[][])
	} body {
		struct Count { /* To force the associative array to behave properly */
			real[movec][movec] c;
		}

		real[movec][movec] ret = 0;
		Count[size_t] count;	/* (4)+2-dimensional array: Markov state = (state of 1 x state of 2 x last move of 1 x last move of 2)
								 * -> aggregate (weighted) counts of move-pairs of the simulation up to this Markov state.
								 * Associative array used as this is really sparse.
								 * As a side bonus, inExpression is a test for past visit */
		static if (!infinite) {
			size_t[] history;	/* Hybridizing the associative array with a simple array of indices for fast indexing (and fast search) */
		}
		size_t mid, mid2;		/* Indices into the Markov chain */
		Move sm, om, sm2, om2;
		static if (weighted) {
			real alphan = 1;	/* Stores alpha^n by successive multiplication */
		}

		static if (minimize) {
			states_minimal; /* Ensure both automata have been state-minimized */
			other.states_minimal;

			const size_t n1_states = st_minimal.length;
			const size_t n2_states = other.st_minimal.length;
		} else {
			const size_t n1_states = states.length;
			const size_t n2_states = other.states.length;
		}
		/* Implementation choice: Markov chain index linearized to avoid double indirection */
		const size_t m_states = n1_states * n2_states * movec * movec;

		size_t index(size_t q1, size_t q2, Move m1, Move m2) { /* Indexing function for state x state x move x move */
			return ((q1 * n2_states + q2) * movec + m1) * movec + m2;
		}

		sm = this.play_init!minimize(); /* Pass on the minimize flag to play_init to simplify code */
		om = other.play_init!minimize();
		mid = index(cur_state, other.cur_state, sm, om);

		count[mid] = Count();
		foreach (ref c; count[mid].c) {
			c[] = 0; /* Initialize for real values */
		}
		count[mid].c[sm][om] = 1; /* At this Markov state, the only move-pair ever played is sm-om (if weighted, at alpha^0) */
		static if (!infinite) {
			history ~= mid; /* Append this Markov state to the past history list */
		}

		static if (infinite) {
			uint[] history; /* Hacked in, unused declaration to allow line below to compile */
		}
		while (infinite || history.length < rounds) { /* Should be optimized away properly */
			sm2 = this.play_next!minimize(om);
			om2 = other.play_next!minimize(sm);
			mid2 = index(cur_state, other.cur_state, sm2, om2);
			if (mid2 in count) { /* The Markov chain has closed the cycle */
				break;
			}

			count[mid2] = count[mid]; /* Copy over the aggregate counts */
			static if (weighted) {
				alphan *= alpha;
				count[mid2].c[sm2][om2] += alphan; /* Weighting by alpha^n */
			} else {
				count[mid2].c[sm2][om2]++; /* And increment the currently played pair */
			}
			static if (!infinite) {
				history ~= mid2; /* Append this state to the past history list */
			}

			sm = sm2;
			om = om2;
			mid = mid2;
		}

		/* If not infinite, the simulation could reach the rounds limit before cycling */
		static if (!infinite) {
			if (history.length == rounds) {
				ret = count[mid2].c; /* Copy last (at rounds-limit) aggregate counts */

				foreach (m; Moves) {
					static if (weighted) {
						/* Normalize total weighted counts, total weighting is sum(n=0..rounds-1)alpha^n = (1-alpha^n)/(1-alpha) */
						ret[m][] *= (1 - alpha) / (1 - alpha ^^ rounds);
					} else {
						ret[m][] /= rounds; /* Normalize total counts */
					}
				}
				return ret;
			}
		}

		/* At this point mid is the last state in the cycle, which closes to mid2 */
		real[movec][movec] temp;
		ret = count[mid2].c; /* Take the accumulated (possibly weighted) moves before the cycle in ret */
		foreach (m1; Moves) {
			foreach (m2; Moves) {
				/* This subtraction counts the (weighted) moves within the cycle only */
				temp[m1][m2] = count[mid].c[m1][m2] - count[mid2].c[m1][m2];
			}
		}
		static if (weighted) { /* Add back the (weighted) cycle-closing move pair */
			alphan *= alpha;
			temp[sm2][om2] += alphan;
		} else {
			temp[sm2][om2]++;
		}

		real cyctot = 0;
		foreach (t1; temp) {
			foreach (t2; t1) {
				cyctot += t2; /* Calculate the total weight of one cycle */
			}
		}
		real total = 0;
		foreach (r1; ret) {
			foreach (r2; r1) {
				total += r2; /* Calculate accumulated weight before the cycle */
			}
		}

		static if (weighted) {
			static if (infinite) {
				if (cyctot > 1e-20) { /* If alpha = 0 or cycle occurs too far ahead, the cycle has negligible weighting */
					/* The total weight (for the complete geometric series) is 1/(1-alpha).
					 * The remainder is the contribution to be made from the cycle, which is thus scaled up to complete the weight */
					const real factor = (1 / (1 - alpha) - total) / cyctot;
					assert(factor >= 1);
					foreach (m1; Moves) {
						foreach (m2; Moves) {
							ret[m1][m2] += factor * temp[m1][m2];
						}
					}
				}

				foreach (ref r1; ret) {
					r1[] *= 1 - alpha; /* Normalize counts */
				}
			} else {
				uint cyclen = 1;
				while (history[$ - cyclen] != mid2) { /* Walk back through the cycle to find its length */
					cyclen++;
				}
				const size_t prefixlen = history.length - cyclen;

				const size_t cyccount = (rounds - prefixlen) / cyclen; /* Number of complete cycles used */
				const size_t endid = rounds - cyccount * cyclen; /* Where the last (possibly complete) cycle ends, one-indexed */
				assert(endid >= prefixlen && endid <= history.length);

				foreach (m1; Moves) {
					foreach (m2; Moves) {
						/* Total weight consists of pre-cycle, cyccount copies of the cycle (finite geometric sum with ratio alpha^cyclen),
						 * plus the tail after the last cycle, weighted down cyccount times cycle length */
						if (endid > 0) {
							ret[m1][m2] += (1 - (alpha ^^ cyclen) ^^ cyccount) / (1 - alpha ^^ cyclen) * temp[m1][m2] +
							  (alpha ^^ cyclen) ^^ cyccount * (count[history[endid - 1]].c[m1][m2] - ret[m1][m2]);
						}  else {
							/* Corner case: cycle starts at very first move and ends right there (tail of cycle is null, so must be subtracted) */
							ret[m1][m2] += (1 - (alpha ^^ cyclen) ^^ cyccount) / (1 - alpha ^^ cyclen) * temp[m1][m2] +
							  (alpha ^^ cyclen) ^^ cyccount * -ret[m1][m2];
						}
					}
				}

				foreach (ref r1; ret) {
					/* Normalize counts, total weighting is sum(n=0..rounds-1)alpha^n = (1-alpha^n)/(1-alpha) */
					r1[] *= (1 - alpha) / (1 - alpha ^^ rounds);
				}
			}
		} else {
			static if (infinite) { /* If infinite horizon, unweighted, simply average over the cycle itself */
				foreach (m1; Moves) {
					foreach (m2; Moves) {
						ret[m1][m2] = temp[m1][m2] / cyctot;
					}
				}
			} else {
				const size_t cyccount = (rounds - to!uint(total)) / roundTo!uint(cyctot); /* Number of complete cycles used */
				const size_t endid = rounds - cyccount * roundTo!uint(cyctot); /* Where the last (possibly complete) cycle ends, one-indexed */
				assert(endid >= total && endid <= history.length);

				foreach (m1; Moves) {
					foreach (m2; Moves) {
						/* The total count of moves is (from start to end of last (complete) cycle) + #complete cycles * per-cycle move-count */
						if (endid > 0) {
							ret[m1][m2] = (count[history[endid - 1]].c[m1][m2] + cyccount * temp[m1][m2]) / rounds;
						} else { /* Corner case: cycle starts at very first move and ends right there (tail of cycle is null and ignored) */
							ret[m1][m2] = cyccount * temp[m1][m2] / rounds;
						}
					}
				}
			}
		}

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

class DFTtourn : GeneN!DFTtourn {
	DFT automaton;
	alias automaton this;

	alias DFT.Move Move;
	alias DFT.Moves Moves;
	alias DFT.movec movec;
	static {
		real[movec][movec] score	= [ [ 3 /* C/opp.C -> R */, 0 /* C/opp.D -> S */ ],
									    [ 5 /* D/opp.C -> T */, 1 /* D/opp.D -> P */ ] ]; /* Payoff matrix */
		uint eval_rounds			= 150;	/* Number of rounds to simulate for fitness evaluation, set to zero or negative for infinite horizon */
		real eval_alpha				= 1;	/* Geometric weighting parameter for fitness evaluation, set to one for unweighted */
		uint eval_rounds_switchover	= 250;	/* Point at which loop-detection algorithm is faster than direct simulation (empirically optimize) */
	}

	/* Restore default constructor */
	this() {}

	/* Direct aliasing copy constructor */
	this(DFT automaton) {
		this.automaton = automaton;
	}

	/* Copy constructor */
	this(const DFTtourn other) {
		automaton = new DFT(other.automaton);
	}

	/* Random initialization constructor */
	this(ref Random rng) {
		automaton = new DFT(rng);
	}

	override string toString() const {
		return to!string(cachefit) ~ ":" ~ to!string(init_state) ~ "/" ~ to!string(init_act) ~ "-" ~ to!string(states);
	}

	override @property DFTtourn dup() const {
		return new DFTtourn(this);
	}

	override void mutate(ref Random rng) {
		automaton.mutate(rng);
	}

	override DFTtourn[2] crossover(const DFTtourn other, ref Random rng) const {
		DFT[2] res = automaton.crossover(other.automaton, rng);
		return [new DFTtourn(res[0]), new DFTtourn(res[1])];
	}

	/* Round-robin tournament on the input.
	 * Compile time option: if statistics, instead of the fitness vector over the input, returns input-population tournament average counts of move-pairs. */
	static real[] tournament(bool statistics = false)(DFTtourn[] input)
	out(ret) {
		static if (statistics) {
			real total = 0;
			foreach (r; ret) {
				total += r;
			}
			assert(approxEqual(total, 1, 1e-15, 1e-15), "tournament!statistics returned vector summing to " ~ to!string(total) ~ " when 1 expected"); //TODO: find sum(real[])

			foreach (m1; Moves) {
				foreach (m2; Moves) {
					assert(approxEqual(ret[m1 * movec + m2], ret[m2 * movec + m1], 1e-15, 1e-15), "tournament!statistics returned vector with entries (" ~ to!string(m1) ~ "," ~ to!string(m2) ~ ") and (" ~ to!string(m2) ~ "," ~ to!string(m1) ~ ") not equal when symmetry expected");
				}
			}
		}
	} body {
		real[] ret;
		real[movec][movec] match;
		real[movec][movec][] results;
		results.length = input.length;
		foreach (ref r1; results) { /* Initialize results array */
			foreach (ref r2; r1) {
				r2[] = 0;
			}
		}

		foreach (i; 0 .. input.length - 1) {
			foreach (j; i + 1 .. input.length) { /* Symmetric game, so run upper triangle for round-robin */
				/* Dispatch the correct simulation function, could be made static */
				if (eval_alpha == 1) { /* If evaluation is unweighted */
					if (eval_rounds == 0) { /* Infinite-horizon */
						match = input[i].simulate_loop!(false,true)(input[j], 0);
					} else if (eval_rounds >= eval_rounds_switchover) { /* Too many rounds, use loop-detection algorithm */
						match = input[i].simulate_loop!(false,false)(input[j], eval_rounds);
					} else {
						match = input[i].simulate!false(input[j], eval_rounds);
					}
				} else {
					/* Number of rounds to cause the tail to drop below 10^-20, approximately machine epsilon on IEEE754 80-bit precision */
					const uint rounds_limit = roundTo!uint(log(10) / log(eval_alpha) * -20);
					if (eval_rounds == 0) {
						if (rounds_limit < eval_rounds_switchover) { /* Easier to directly simulate until tail arithmetically negligible */
							match = input[i].simulate!true(input[j], rounds_limit, eval_alpha);
						} else {
							match = input[i].simulate_loop!(true,true)(input[j], 0, eval_alpha);
						}
					} else if (eval_rounds > rounds_limit) { /* Cap the number of rounds used at limit */
						if (rounds_limit < eval_rounds_switchover) { /* Easier to directly simulate until tail arithmetically negligible */
							match = input[i].simulate!true(input[j], rounds_limit, eval_alpha);
						} else {
							match = input[i].simulate_loop!(true,true)(input[j], 0, eval_alpha); /* Infinite horizon is easier to handle */
						}
					} else {
						if (eval_rounds >= eval_rounds_switchover) { /* Too many rounds, use loop-detection algorithm */
							match = input[i].simulate_loop!(true,false)(input[j], eval_rounds, eval_alpha);
						} else {
							match = input[i].simulate!true(input[j], eval_rounds, eval_alpha);
						}
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

		static if (statistics) {
			ret.length = movec * movec; /* For signature compatibility, the move-pair matrix is flattened; it should be symmetric anyway */
			ret[] = 0;

			foreach (i; 0 .. input.length) {
				foreach (m1; Moves) {
					foreach (m2; Moves) {
						ret[m1 * movec + m2] += results[i][m1][m2];
					}
				}
			}
			/* Normalize to input-population average: nC2 matches, then results are double counted per match */
			ret[] /= input.length * (input.length - 1);

			return ret;
		} else {
			ret.length = input.length;
			ret[] = 0;

			foreach (i; 0 .. input.length) {
				foreach (m1; Moves) {
					foreach (m2; Moves) {
						ret[i] += results[i][m1][m2] * score[m1][m2]; /* Total score is dot product of result-type count and score matrix */
					}
				}
			}
			ret[] /= input.length - 1; /* Normalize to per-round, per-match scores */

			foreach (i; 0 .. input.length) {
				input[i].cachefit = ret[i]; /* Store fitness results for participating automata */
			}
			return ret;
		}
	}
}

class DFTmatch : GeneL!DFTmatch {
	DFT automaton;
	alias automaton this;

	alias DFT.Move Move;
	alias DFT.Moves Moves;
	alias DFT.movec movec;
	static {
		real alpha					= 0.8;	/* Fingerprint evaluation alpha geometric parameter */
		int target_subdiv_depth		= 2;	/* 4 ^^ depth subsquares (4 ^^ (depth + 1) points) in the target profile */
		real[8][] target_profile;			/* Target profile (fingerprint */
	}

	/* Restore default constructor */
	this() {}

	/* Direct aliasing copy constructor */
	this(DFT automaton) {
		this.automaton = automaton;
	}

	/* Copy constructor */
	this(const DFTmatch other) {
		automaton = new DFT(other.automaton);
	}

	/* Random initialization constructor */
	this(ref Random rng) {
		automaton = new DFT(rng);
	}

	override string toString() const {
		return to!string(cachefit) ~ ":" ~ to!string(init_state) ~ "/" ~ to!string(init_act) ~ "-" ~ to!string(states);
	}

	override @property DFTmatch dup() const {
		return new DFTmatch(this);
	}

	override void mutate(ref Random rng) {
		automaton.mutate(rng);
	}

	override DFTmatch[2] crossover(const DFTmatch other, ref Random rng) const {
		DFT[2] res = automaton.crossover(other.automaton, rng);
		return [new DFTmatch(res[0]), new DFTmatch(res[1])];
	}

	static setTarget(DFT target) {
		target_profile.length = 4 << (target_subdiv_depth * 2);

		void target_set(real y, real y2, real z, real z2, size_t offset, int depth) {
			immutable real cp = 1 + sqrt(1.0 / 3);
			immutable real cn = 1 - sqrt(1.0 / 3);
			if (depth) {
				target_set(y, (y + y2) / 2, z, (z + z2) / 2, offset, depth - 1);
				target_set((y + y2) / 2, y2, z, (z + z2) / 2, offset + (1 << (depth * 2)), depth - 1);
				target_set(y, (y + y2) / 2, (z + z2) / 2, z2, offset + (2 << (depth * 2)), depth - 1);
				target_set((y + y2) / 2, y2, (z + z2) / 2, z2, offset + (3 << (depth * 2)), depth - 1);
				return;
			}
			target_profile[offset] = target.fp_point((y * cn + y2 * cp) / 2, (z * cn + z2 * cp) / 2, alpha);
			target_profile[offset + 1] = target.fp_point((y * cp + y2 * cn) / 2, (z * cn + z2 * cp) / 2, alpha);
			target_profile[offset + 2] = target.fp_point((y * cn + y2 * cp) / 2, (z * cp + z2 * cn) / 2, alpha);
			target_profile[offset + 3] = target.fp_point((y * cp + y2 * cn) / 2, (z * cp + z2 * cn) / 2, alpha);
		}

		target_set(0, 1, 0, 1, 0, target_subdiv_depth);
	}

	override real evaluate()
	in {
		assert(target_profile.length == 4 << target_subdiv_depth, "Inconsistent target_profile length: " ~ to!string(target_profile.length) ~ ", " ~ to!string(4 << (target_subdiv_depth * 2)) ~ " required for target_subdiv_depth " ~ to!string(target_subdiv_depth));
	} body {
		/* Single-point fingerprint distance between this automaton and target profile, unscaled for speed */
		real fp_diffpt(real y, real z, const real[8] target) {
			real[8] x = automaton.fp_point(y, z, alpha);
			x[] -= target[];
			foreach (i; 0 .. 4) {
				if ((x[i] < 0) ^ (x[i + 4] < 0)) {
					x[i] = fabs((x[i] * x[i] + x[i + 4] * x[i + 4]) / (x[i] - x[i + 4]));
				} else {
					x[i] = fabs(x[i] + x[i + 4]);
				}
			}
			return (x[0] + x[1]) + (x[2] + x[3]);
		}

		/* Fingerprint integration by recursive divide-and-conquer with 3rd order product Gauss cubature, unscaled for speed
		 * Uses fixed, precomputed target_profile for second fingerprint */
		real fp_integrate(real y, real y2, real z, real z2, const real[8][] target_slice, int depth)
		in {
			assert(target_slice.length == 4 << (depth * 2));
		} body {
			immutable real cp = 1 + sqrt(1.0 / 3);
			immutable real cn = 1 - sqrt(1.0 / 3);
			if (depth) {
				const size_t sl = target_slice.length / 4;
				return (fp_integrate(y, (y + y2) / 2, z, (z + z2) / 2, target_slice[0 .. sl], depth - 1) +
				  fp_integrate((y + y2) / 2, y2, z, (z + z2) / 2, target_slice[sl .. sl * 2], depth - 1)) +
				  (fp_integrate(y, (y + y2) / 2, (z + z2) / 2, z2, target_slice[sl * 2 .. sl * 3], depth - 1) +
				  fp_integrate((y + y2) / 2, y2, (z + z2) / 2, z2, target_slice[sl * 3 .. $], depth - 1));
			}
			return (fp_diffpt((y * cn + y2 * cp) / 2, (z * cn + z2 * cp) / 2, target_slice[0]) +
			  fp_diffpt((y * cp + y2 * cn) / 2, (z * cn + z2 * cp) / 2, target_slice[1])) +
			  (fp_diffpt((y * cn + y2 * cp) / 2, (z * cp + z2 * cn) / 2, target_slice[2]) +
			  fp_diffpt((y * cp + y2 * cn) / 2, (z * cp + z2 * cn) / 2, target_slice[3]));
		}

		return ldexp(fp_integrate(0, 1, 0, 1, target_profile, target_subdiv_depth), -2 * target_subdiv_depth - 4);
	}
}

