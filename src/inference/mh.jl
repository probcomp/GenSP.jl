# Version of Gen's Meteropolis-Hastings algorithm that accepts a proposal that is a
# distribution over choicemaps, rather than a generative function.
function Gen.metropolis_hastings(trace::Gen.Trace, proposal::Distribution{ChoiceMap}, proposal_args::Tuple)
    model_args = Gen.get_args(trace)
    argdiffs = map((_)->Gen.NoChange(), model_args)

    _, fwd_weight, choices = Gen.propose(proposal, (trace, proposal_args...))
    new_trace, weight, _, discard = Gen.update(trace, model_args, argdiffs, choices)
    bwd_weight = Gen.assess(proposal, proposal_args, discard)
    alpha = weight - fwd_weight + bwd_weight
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        return (trace, false)
    end
end