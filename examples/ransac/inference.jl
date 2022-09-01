@gen function ransac_proposal(xs, ys, K)
    params ~ ransac_safe(xs, ys, K)
end