module MagNet

include("Radon/interpolation.jl")
include("Radon/warp.jl")
include("Radon/Radon.jl")

export inbounds, bilinear_interpolation, trilinear_interpolation, around_points,
            Euler, tilt, warp, tilt_vecfld, radon, radon_3d_object, radon_vecfld

#= using CUDA

#max_ava_threads = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)

@inline function cudims(n::Integer, N::Integer=1024)
    threads = min(n, N)
    return ceil(Int, n / threads), threads
end
  
cudims(a::AbstractArray) = cudims(length(a))

include("CuRadon/kernels.jl")
include("CuRadon/tilt.jl")
include("CuRadon/CuRadon.jl") =#

include("util.jl")
export FT, IFT

include("curl.jl")
export grad, curl, divergence

include("padding.jl")
export center_crop, center_padding, complex2real

include("vector_potential.jl")
export compute_magnetic_vector_potential

include("subsampling.jl")
export drawline_radial, radial_subsample_XY

include("projection.jl")
export project_XY


include("load_image.jl")
export load_image_x, load_image_y

end # module
