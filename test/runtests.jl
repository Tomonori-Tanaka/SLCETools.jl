using Test

const TEST_MODE = get(ENV, "TEST_MODE", "default")

@testset "SCETools.jl" begin
    if TEST_MODE in ("default", "all", "unit")
        include("unit/test_site_engine.jl")
        include("unit/test_mfa_sampler.jl")
        include("unit/test_exchange.jl")
        include("unit/test_tensorial.jl")
        include("unit/test_multipole.jl")
        include("unit/test_vasp_incar.jl")
    end
    if TEST_MODE in ("default", "all", "aqua")
        include("aqua.jl")
    end
    if TEST_MODE in ("all", "jet")
        include("jet.jl")
    end
end
