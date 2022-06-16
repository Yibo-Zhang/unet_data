using JuMag
using Printf

function rebuild_sim(sim)
    ms = 1e5*(1+7*rand())
    A = 2e-12
    ku = 1e3*(1+9*rand())
    D = 5e-5*(1+2*rand())


    sim.interactions = filter!(x->x.name=="demag",sim.interactions)

    headers = ["step", "E_total", ("m_x", "m_y", "m_z")]
    units = ["<>", "<J>",("<>", "<>", "<>")]
    results = [o::JuMag.AbstractSim -> o.saver.nsteps,
            o::JuMag.AbstractSim -> sum(o.energy),  
            JuMag.average_m]
    sim.saver = JuMag.DataSaver(string("sim", ".txt"), 0.0, 0, false, headers, units, results)

    set_Ms(sim, ms)
    add_exch(sim, A)


    if rand() > 0.7
        add_anis(sim, ku)
    end

    if rand() > 0.15
        add_dmi(sim, D)
    end

    init_m0_random(sim)
end

function gen_mag_files(ids, n::Int; d=5e-9)
    JuMag.cuda_using_double(false)
    folder = "npys/m"
    if !isdir(folder)
        mkpath(folder)
    end


    mesh = FDMeshGPU(nx=n,ny=n,nz=n,dx=d,dy=d,dz=d)
    sim = Sim(mesh,driver="SD",save_data=true)
    set_Ms(sim,1e5)
    add_demag(sim)

    for id in ids
        rebuild_sim(sim)
        relax(sim, maxsteps=20000, stopping_dmdt=1)
        m = Array(sim.spin)
        npzwrite(@sprintf("%s/%d.npy",folder,id), m)
    end

    # remove trash
    # rm("sim.txt")
end