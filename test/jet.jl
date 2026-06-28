using SCETools
using JET

@testset "JET" begin
    JET.test_package(SCETools; target_modules = (SCETools,))
end
