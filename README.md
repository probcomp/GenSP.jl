# GenSP.jl

### Distributions in Gen and GenSP
In Gen, a `Distribution{T}` is a specific kind of generative function that stores a single
value in its `ChoiceMap`. That value is also the return value of the Distribution.
Gen provides default implementations of the GFI for new subtypes of `Distribution{T}`
(like `Normal`)
that implement the distribution interface (`random`, `logpdf`, `is_discrete`, and other
methods for gradient-based inference).

GenSP introduces a new subtype `SPDistribution{T} <: Distribution{T}`. Users can declare
new subtypes of `SPDistribution{T}` and instead of implementing `random` and `logpdf`,
implement `random_weighted` and `estimate_logpdf`.

### Inference in GenSP
GenSP exposes a new library for inference. It works a lot like Gen's inference library,
with the following key differences:

* In GenSP, inference algorithms are themselves generative functions. In particular, 
GenSP's inference algorithms are `SPDistribution{ChoiceMap}`s 
that take as input a `Target` posterior and produce as output
a `ChoiceMap` approximately sampled from the posterior. 
(`Target` is a struct type that GenSP exposes, 
wrapping together a generative function, arguments to it,
and a `ChoiceMap` of observations.) Because inference algorithms
are `SPDistribution`s, they can estimate their own output densities.

* Instead of proposal *generative functions*, GenSP inference methods accept
proposal *distributions* -- distributions over `ChoiceMap`s containing the
unconstrained choices of the target. Such distributions can be obtained from
generative functions by using the `ChoiceMapDistribution` combinator. That
combinator can also be used to marginalize auxiliary variables from proposals,
by selecting only the choices meant to serve as the proposal.

### Marginalization
New distributions can be created by marginalizing generative functions:

* `Marginal{T}(gen_fn, inf_alg, addr)` -- the marginal distribution of the choice at address `addr` in `gen_fn`.
* `ChoiceMapDistribution(gen_fn, selection=AllSelection(), inf_alg=default_importance(1))` -- the marginal distribution of `selection` under `gen_fn`.




