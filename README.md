# GenProx.jl

### Distributions in Gen and Prox
In Gen, a `Distribution{T}` is a specific kind of generative function that stores a single
value in its `ChoiceMap`. That value is also the return value of the Distribution.
Gen provides default implementations of the GFI for new subtypes of `Distribution{T}`
(like `Normal`)
that implement the distribution interface (`random`, `logpdf`, `is_discrete`, and other
methods for gradient-based inference).

Prox introduces a new subtype `ProxDistribution{T} <: Distribution{T}`. Users can declare
new subtypes of `ProxDistribution{T}` and instead of implementing `random` and `logpdf`,
implement `random_weighted` and `estimate_logpdf`.

### Inference in Prox
Prox exposes a new library for inference. It works a lot like Gen's inference library,
with the following key differences:

* In Prox, inference algorithms are themselves generative functions. In particular, 
Prox's inference algorithms are `ProxDistribution{ChoiceMap}`s 
that take as input a `Target` posterior and produce as output
a `ChoiceMap` approximately sampled from the posterior. 
(`Target` is a struct type that Prox exposes, 
wrapping together a generative function, arguments to it,
and a `ChoiceMap` of observations.) Because inference algorithms
are `ProxDistribution`s, they can estimate their own output densities.

* Instead of proposal *generative functions*, Prox inference methods accept
proposal *distributions* -- distributions over `ChoiceMap`s containing the
unconstrained choices of the target. Such distributions can be obtained from
generative functions by using the `ChoiceMapDistribution` combinator. That
combinator can also be used to marginalize auxiliary variables from proposals,
by selecting only the choices meant to serve as the proposal.

### Marginalization
New distributions can be created by marginalizing generative functions:

* `Marginal{T}(gen_fn, inf_alg, addr)`
* `ChoiceMapDistribution(gen_fn, selection=AllSelection(), inf_alg=default_importance(1))`




