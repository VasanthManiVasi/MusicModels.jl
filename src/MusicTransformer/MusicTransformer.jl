module MusicTransformer

using Flux
using Transformers

include("layers.jl")
include("models.jl")
include("musicencoders.jl")
include("generate.jl")
include("pretrain.jl")

end
