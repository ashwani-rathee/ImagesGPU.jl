using TestImages, ColorTypes, FixedPointNumbers,FileIO, Pkg
Pkg.activate(".")
using CuImages
img = testimage("mandrill")
img_gpu = Cu(img)
