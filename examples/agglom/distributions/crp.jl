# CRP as a distribution over partitions.
struct CRP <: Distribution{Set{Set{Int}}} end
const crp = CRP()

function Gen.random(::CRP, n, alpha)
    # Generate a random partition of the set {1, ..., n}
    # according to the Chinese restaurant process.
    # alpha is the concentration parameter.
    # The partition is returned as a set of sets.
    # The sets are disjoint.

    # Track the restaurant's tables.
    tables = Vector{Set{Int}}()

    for i in 1:n
        # Choose a table according to the CRP.
        # The probability of a table is proportional to its size.
        # The probability of starting a new table is proportional to alpha.
        unnormalized_table_probs = [[length(t) for t in tables]..., alpha]
        table_probs = unnormalized_table_probs / sum(unnormalized_table_probs)
        table = categorical(table_probs)
        if (table > length(tables))
            # Start a new table.
            push!(tables, Set(i))
        else
            # Add the customer to an existing table.
            push!(tables[table], i)
        end
    end

    # Return the partition -- a set of tables, not a vector.
    return Set(tables)
end

function Gen.logpdf(::CRP, partition::Set{Set{Int}}, n, alpha)
    # Iterate over the tables in the partition:
    customer = 0
    logprob = 0.0
    for table in partition
        table_size = length(table)

        # The first customer starts a new table.
        logprob += log(alpha) - log(customer + alpha)
        customer += 1

        # The others join the table.
        for others_at_table in 1:table_size-1
            logprob += log(others_at_table) - log(customer + alpha)
            customer += 1
        end
    end
    return logprob
end
