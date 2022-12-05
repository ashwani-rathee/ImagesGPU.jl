module CuImages

using CUDA
using FixedPointNumbers
using ColorTypes
using TensorCore
using KernelAbstractions
using CUDAKernels
using ImageFiltering

const GpuBackend = CUDADevice()

include("augmentation.jl")
include("enhance.jl")
include("filtering.jl")
include("utils.jl")

# from utils
export Cu
export Ar

# from augmentation
export rotate
export scale
export translate

# for filtering
export gaussian
export sobel
export prewitt
export ando3
export scharr
export bickley




# convert image to gpu array 
# do operation on gpu
# at end of algorithm return normal array back 




end # module
