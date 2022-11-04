using NearestNeighbors

@gen function render_point_cloud_exact(X, sigma, n)
    cloud ~ Map(render_point_exact)(fill(X, n), fill(sigma, n))
    return cloud
end

@gen function render_point_cloud_safe(X, sigma, n)
    tree = NearestNeighbors.KDTree(X)
    cloud ~ Map(render_point_safe)(fill(X, n), fill(tree, n), fill(sigma, n))
    return cloud
end

@gen function render_point_cloud_unsafe(X, sigma, n)
    tree = NearestNeighbors.KDTree(X)
    cloud ~ Map(render_point_unsafe)(fill(X, n), fill(tree, n), fill(sigma, n))
    return cloud
end


