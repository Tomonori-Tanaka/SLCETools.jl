using SLCETools
using JET

@testset "JET" begin
    JET.test_package(SLCETools; target_modules = (SLCETools,))
end
