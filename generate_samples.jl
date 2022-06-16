push!(LOAD_PATH, "../../src")
using MagNet
using NPZ
using Printf
using FFTW
using JuMag

include("run_JuMag.jl")

function gen_random_angles(;c1=6,c2=14)
    """
    generate random subsampling angles for function radial_subsampling
    """
    c = rand(c1:c2)
    angles = (rand(c) .- 0.5).* (2*pi/3)
    if !(0.0 in angles)
        append!(angles, 0.0)
    end

    return angles
end

function generate_sample(id,n,N)
    m = npzread("npys/m/$id.npy")
    m = reshape(m, 3,n,n,n)
    A = compute_magnetic_vector_potential(m, N=2*N)
    npzwrite(@sprintf("npys/A/%d.npy",id), A)

    A_shift = ifftshift(A, (2,3,4))
    A_k = fft(A_shift, (2,3,4))
    A_k_c = fftshift(A_k, (2,3,4))

    output_data = complex2real(A_k_c, N)
    npzwrite("npys/output_data/$id.npy", permutedims(output_data, (2,3,4,1)))

    alphas = gen_random_angles()
    betas = gen_random_angles()
    projection_XY = project_XY(A_k_c)
    input_data = radial_subsample_XY(projection_XY, alphas, betas)

    input_data = complex2real(input_data, N)
    npzwrite("npys/input_data/$id.npy", permutedims(input_data, (2,3,4,1)))
end

function generate_sample_with_noise(id,n,N)
    m = npzread("npys/m/$id.npy")
    m = reshape(m,3,n,n,n)
    A = compute_magnetic_vector_potential(m, N=2*N)
    npzwrite(@sprintf("npys/A/%d.npy",id), A)

    # noised sample
    A_noise = add_gauss(A, 0.1)
    A_noise_shift = ifftshift(A_noise, (2,3,4))
    A_noise_k = fft(A_noise_shift, (2,3,4))
    A_k_c_noise = fftshift(A_noise_k, (2,3,4))
    # 在这里 分开，在 fft 之前产生 加noise

    # unnoised sample
    A_shift = ifftshift(A, (2,3,4))
    A_k = fft(A_shift, (2,3,4))
    A_k_c = fftshift(A_k, (2,3,4))

    output_data = complex2real(A_k_c, N)
    npzwrite("npys/output_data/$id.npy", permutedims(output_data, (2,3,4,1)))


    return A_k_c,A_k_c_noise
end

function generate_sample_with_noise_in_K_space(id,n,N)
    m = npzread("npys/m/$id.npy")
    m = reshape(m, 3,n,n,n)
    A = compute_magnetic_vector_potential(m, N=2*N)
    npzwrite(@sprintf("npys/A/%d.npy",id), A)

    A_shift = ifftshift(A, (2,3,4))
    A_k = fft(A_shift, (2,3,4))
    A_k_c = fftshift(A_k, (2,3,4))

    output_data = complex2real(A_k_c, N)
    npzwrite("npys/output_data/$id.npy", permutedims(output_data, (2,3,4,1)))

    alphas = gen_random_angles()
    betas = gen_random_angles()
    projection_XY = project_XY(A_k_c)

    input_data = radial_subsample_XY(projection_XY, alphas, betas)
    input_data = complex2real(input_data, N)
    npzwrite("npys/input_data/$id.npy", permutedims(input_data, (2,3,4,1)))


    noised_input_data = radial_subsample_XY_with_noise(projection_XY, alphas, betas)
    noised_input_data = complex2real(noised_input_data, N)
    npzwrite("npys/input_data/noise/$id.npy", permutedims(noised_input_data, (2,3,4,1)))
end


function main(ids)
    d, n, N = 5e-9, 80, 96

    for f in ["npys/m", "npys/A", "npys/input_data","npys/input_data/noise", "npys/output_data"]
        !isdir(f) && mkpath(f)
    end

    gen_mag_files(ids, n, d=d)
    for i in ids
        generate_sample_with_noise_in_K_space(i,n,N)
    end
end
main(1:4)
