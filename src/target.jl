# Represents an unnormalized target distribution as a pairing of a 
# generative function (with specific arguments) and a choicemap of constraints.
struct Target
    p :: GenerativeFunction
    args :: Tuple
    constraints :: ChoiceMap
end

export Target