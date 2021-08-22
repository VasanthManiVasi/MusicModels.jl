using Flux: loadparams!

import ..loading_method

loading_method(::Val{:perfrnn}) = load_perfrnn

function load_perfrnn(weights, config)
    lstm_units = config["lstm_units"]
    input_dims = config["input_dims"]

    layers = []
    for i in 1:config["num_layers"]
        layer = BasicLSTM(
                     (i == 1) ? input_dims : lstm_units,
                     lstm_units
                )
        push!(layers, layer)
    end
    push!(layers, Dense(lstm_units, input_dims))

    model = Chain(layers...)

    weight_names = keys(weights)
    rnn_weights = filter(name -> occursin("rnn", name), weight_names)
    dense_weights = filter(name -> occursin("fully_connected", name), weight_names)

    for i in 1:length(rnn_weights)
        cell_weights = filter(name -> occursin("cell_$(i-1)", name), rnn_weights)
        for j in cell_weights
            if occursin("kernel", j)
                kernel = weights[j]
                if i == 1
                    Wi = @view kernel[:, 1:input_dims]
                    Wh = @view kernel[:, input_dims+1:end]
                else
                    Wi = @view kernel[:, 1:lstm_units]
                    Wh = @view kernel[:, lstm_units+1:end]
                end
                loadparams!(model[i], [Wi, Wh])
            elseif occursin("bias", j)
                loadparams!(model[i].cell.b, [weights[j]])
            end
        end
    end

    for j in dense_weights
        if occursin("weights", j)
            loadparams!(model[end].W, [weights[j]])
        elseif occursin("biases", j)
            loadparams!(model[end].b, [weights[j]])
        end
    end

    num_velocitybins = config["velocity_bins"]
    perfencoder = PerformanceOneHotEncoding(num_velocitybins=num_velocitybins)
    perfrnn = PerfRNN(model, perfencoder, num_velocitybins)
end