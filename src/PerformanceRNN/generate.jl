export PerfRNN, generate

using StatsBase
using Flux: gate

import ..MusicTransformer: generate

struct PerfRNN
    model
    encoder::PerformanceOneHotEncoding
    num_velocitybins::Int
end

"""
    generate(perfrnn::PerfRNN, performance::Performance)

Generate `PerformanceEvent`s by sampling from a PerformanceRNN model.
"""
function generate(perfrnn::PerfRNN;
                  primer::Performance=Performance(100, velocity_bins=perfrnn.num_velocitybins),
                  numsteps=3000,
                  as_notesequence=false)

    model, encoder = perfrnn.model, perfrnn.encoder
    Flux.reset!(model)

    performance = deepcopy(primer)

    if isempty(performance)
        push!(performance, encoder.defaultevent)
    end

    # Primer is already numsteps or longer
    if performance.numsteps >= numsteps
        return performance
    end

    indices = map(event -> encode_event(event, encoder), performance)
    inputs = map(index -> Flux.onehot(index, encoder.labels), indices)

    outputs = model.(inputs)
    out = wsample(encoder.labels, softmax(outputs[end]))
    push!(performance, decode_event(out, encoder))

    while performance.numsteps < numsteps
        input = Flux.onehot(out, encoder.labels)
        logits = model(input)
        out = wsample(encoder.labels, softmax(logits))
        push!(performance, decode_event(out, encoder))
    end

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    midifile(ns)
end
