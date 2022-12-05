# CuImages

This is basically a playground for developing the GPU algorithms and support differentiable
programming for image processing in Julia. The goal is to provide a set of GPU algorithms
for image processing in Julia. The algorithms are implemented using CUDA.jl,
CUDAKernels.jl and KernelAbstractions.jl

We want to currently focus on following topics:
- Augmentation
- Enchancement(adjust gamma, histogramequalisation, etc etc)
- Image Filtering

### Installation:

```julia
julia> add https://github.com/ashwani-rathee/CuImages.jl.git

julia> using CuImages
```

### Usage:

Focus of this package currently is on following operations:
- Convert Image to CuArray
- Apply Algorithm on it
- Convert CuArray to Image

Point of this current package is to learn:
- how to use CUDA.jl
- how to use CUDAKernels.jl
- how to use KernelAbstractions.jl
- Designing GPU algorithms, GPU kernels, CPU fallbacks and Parallel GPU computing
- Benchmark and learn what's fast
- Learn about the design choices that need to be made for GPU support and differentiable computer vision support

Now, how to convert AbstractArray to CuArray and vice versa:

```julia    
julia> using CuImages, TestImages

julia> img = testimage("cameraman");

julia> img_gpu = Cu(img);

julia> img_cpu = Ar(img_gpu);
```

Now, how to apply algorithm on it:

```julia
julia> img_gpu = Cu(img);

julia> img_res = scale(img_gpu, 0.5, 0.2);

julia> img_res_cpu = Ar(img_res);
```

Using CuImages.jl, you can apply following algorithms on image.

For Augmentation, below mentioned algorithms are available:
- Scale Operation
- Rotate Operation
- Translate Operation

For Filtering, below mentioned algorithms are available:
- Gaussian Filter
- Sobel Filter
- Prewitt Filter
- Ando3 Filter
- Ando5 Filter
- Scharr Filter
- Bickley Filter

### Contributions and Issues:

If you have questions about GIFImages.jl, feel free to get in touch via Slack or open an issue hearts