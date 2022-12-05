
# cropping

# flipping x and y

# scaling
@kernel function scale_kernel(out, inp, scale)
    x_idx, y_idx = @index(Global, NTuple)

    x_outidx = floor(Int, x_idx * scale[1])
    y_outidx = floor(Int, y_idx * scale[2])

    if (1 <= x_outidx <= size(out,1)) && (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
end
     

function scale(img::CuArray, x::Float64=0.5, y::Float64=0.2)
    img_new = similar(img)
    img_new .= RGB(0)
    wait(scale_kernel(GpuBackend)(img_new, img, (x, y); ndrange=size(img), workgroupsize=(32,32)))
    return img_new
end   

# rotating
@kernel function rotate_kernel(out, inp, angle)
    x_idx, y_idx = @index(Global, NTuple)

    x_centidx = x_idx - (size(inp,1)÷2)
    y_centidx = y_idx - (size(inp,2)÷2)
    x_outidx = round(Int, (x_centidx*cos(angle)) + (y_centidx*-sin(angle)))
    y_outidx = round(Int, (x_centidx*sin(angle)) + (y_centidx*cos(angle)))
    x_outidx += (size(inp,1)÷2)
    y_outidx += (size(inp,2)÷2)

    if (1 <= x_outidx <= size(out,1)) &&
       (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
end
     
function rotate(img::CuArray, x::Int)
    img_new = similar(img)
    img_new .= RGB(0)
    wait(rotate_kernel(GpuBackend)(img_new, img, deg2rad(x); ndrange=size(img), workgroupsize=(32,32)))
    return img_new
end  

#translation
@kernel function translate_kernel(out, inp, translation)
    x_idx, y_idx = @index(Global, NTuple)

    x_outidx = x_idx + translation[1]
    y_outidx = y_idx + translation[2]

    if (1 <= x_outidx <= size(out,1)) && (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
end

function translate(img::CuArray,x::Int,y::Int)
    img_new = similar(img)
    img_new .= RGB(0)
    wait(translate_kernel(GpuBackend)(img_new, img, (x, y); ndrange=size(img), workgroupsize=(32,32)))
    return img_new
end
