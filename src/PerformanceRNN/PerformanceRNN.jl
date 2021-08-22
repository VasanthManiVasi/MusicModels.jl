module PerformanceRNN

using Flux
using MIDI
using NoteSequences
using NoteSequences.PerformanceRepr

include("layers.jl")
include("generate.jl")
include("pretrain.jl")

end
