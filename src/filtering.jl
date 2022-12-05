# corelation kernel
@kernel function corr_kernel(out, inp, kern, offsets)
    x_idx, y_idx = @index(Global, NTuple)

    out_T = eltype(out)

    if (1 <= x_idx <= size(out,1)) && (1 <= y_idx <= size(out,2))
        x_toff, y_toff = offsets

        # create our accumulator
        acc = zero(out_T)

        # iterate in column-major order for efficiency
        for y_off in -y_toff:1:y_toff, x_off in -x_toff:1:x_toff
            y_inpidx, x_inpidx = y_idx+y_off, x_idx+x_off
            if (1 <= y_inpidx <= size(inp,2)) && (1 <= x_inpidx <= size(inp,1))
                y_kernidx, x_kernidx = y_off+y_toff+1, x_off+x_toff+1
                acc += hadamard(inp[x_inpidx, y_inpidx],
                                kern[x_kernidx, y_kernidx])
            end
        end
        out[x_idx, y_idx] = acc
    end
end
     

# gaussian
function gaussian(img::CuArray)
    gaussian_k = Kernel.gaussian(3)
    gaussian_kG = CuArray(map(x->RGB{Float32}(Gray(x)), gaussian_k.parent))
    gaussian_offsets = abs.(gaussian_k.offsets) .- 1
    
    img_new = similar(img)
    img_new .= RGB(0)
    wait(corr_kernel(GpuBackend)(img_new, img, gaussian_kG, gaussian_offsets; ndrange=size(img), workgroupsize=(16,16)))
    return img_new
end

# laplacian


# sobel
function sobel(img::CuArray)
    sobel_k = Kernel.sobel()
    sobel_kG = CuArray(map(RGB{Float32}, sobel_k[1].parent))
    sobel_offsets = abs.(sobel_k[1].offsets) .- 1

    img_new = similar(img)
    wait(corr_kernel(GpuBackend)(img_new, img, sobel_kG, sobel_offsets; ndrange=size(img), workgroupsize=(16,16)))
    # Post-process the Sobel gradients into something comprehendable by humans
    img_new = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), img_new)
    return img_new
end
     
#prewitt
function prewitt(img::CuArray)
    prewitt_k = Kernel.prewitt()
    prewitt_kG = CuArray(map(RGB{Float32}, prewitt_k[1].parent))
    prewitt_offsets = abs.(prewitt_k[1].offsets) .- 1

    img_new = similar(img)
    wait(corr_kernel(GpuBackend)(img_new, img, prewitt_kG, prewitt_offsets; ndrange=size(img), workgroupsize=(16,16)))
    # Post-process the Sobel gradients into something comprehendable by humans
    img_new = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), img_new)
    return img_new
end

# ando3
function ando3(img::CuArray)
    ando3_k = Kernel.ando3()
    ando3_kG = CuArray(map(RGB{Float32}, ando3_k[1].parent))
    ando3_offsets = abs.(ando3_k[1].offsets) .- 1

    img_new = similar(img)
    wait(corr_kernel(GpuBackend)(img_new, img, ando3_kG, ando3_offsets; ndrange=size(img), workgroupsize=(16,16)))
    # Post-process the Sobel gradients into something comprehendable by humans
    img_new = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), img_new)
    return img_new
end


# scharr
function scharr(img::CuArray)
    scharr_k = Kernel.scharr()
    scharr_kG = CuArray(map(RGB{Float32}, scharr_k[1].parent))
    scharr_offsets = abs.(scharr_k[1].offsets) .- 1

    img_new = similar(img)
    wait(corr_kernel(GpuBackend)(img_new, img, scharr_kG, scharr_offsets; ndrange=size(img), workgroupsize=(16,16)))
    # Post-process the Sobel gradients into something comprehendable by humans
    img_new = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), img_new)
    return img_new
end

# bickley
function bickley(img::CuArray)
    bickley_k = Kernel.bickley()
    bickley_kG = CuArray(map(RGB{Float32}, bickley_k[1].parent))
    bickley_offsets = abs.(bickley_k[1].offsets) .- 1

    img_new = similar(img)
    wait(corr_kernel(GpuBackend)(img_new, img, bickley_kG, bickley_offsets; ndrange=size(img), workgroupsize=(16,16)))
    # Post-process the Sobel gradients into something comprehendable by humans
    img_new = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), img_new)
    return img_new
end

