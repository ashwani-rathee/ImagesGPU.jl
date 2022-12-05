# help convert images to gpu arrays
function Cu(img::AbstractArray)
    type = nameof(typeof(img[1]))
    return CuArray(map(eval(type){Float32}, img))
end

# help convert gpu arrays to images

function Ar(img::CuArray)
    n_img = Array(img)
    type = nameof(typeof(img[1]))
    return eval(type){N0f8}.(n_img)
end
