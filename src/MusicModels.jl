module MusicModels

using Flux
using CUDA

device = Flux.cpu

if CUDA.has_cuda()
    device = Flux.gpu
    println("GPU is enabled.")
    CUDA.allowscalar(false)
end

include("pretrain.jl")
include("datasets.jl")
include("MusicTransformer/MusicTransformer.jl")
include("PerformanceRNN/PerformanceRNN.jl")

using .PerformanceRNN
using .MusicTransformer

export generate

function call_registers()
    register_configs(pretrained_configs)
    register_datasets(dataset_configs)
end

@init call_registers()

end
