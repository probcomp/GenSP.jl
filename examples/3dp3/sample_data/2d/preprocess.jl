using Images

function point_cloud_from_image(image)
    if image isa String
        image = load(image)
    end

    HEIGHT, WIDTH = size(image)

    # Create a list of (x, y) coordinates that have alpha > 0.
    points = []
    for y in 1:HEIGHT
        for x in 1:WIDTH
            if image[y, x].alpha > 0
                push!(points, [x - WIDTH / 2, y - HEIGHT / 2])
            end
        end
    end
    return hcat(points...)
end

function image_from_point_cloud(cloud, size)
    HEIGHT, WIDTH = size
    image = zeros(RGBA, (HEIGHT, WIDTH))
    cloud = Set([(x, y) for (x, y) in eachcol(cloud)])
    for y in 1:HEIGHT
        for x in 1:WIDTH
            image[y, x] = RGBA(0, 0, 0, (x - WIDTH / 2, y - HEIGHT / 2) in cloud ? 1 : 0)
        end
    end
    return image
end

clouds = [point_cloud_from_image("examples/3dp3/sample_data/2d/$i.png") for i in 1:5];
cloud_obs = point_cloud_from_image("examples/3dp3/sample_data/2d/drawing.png");