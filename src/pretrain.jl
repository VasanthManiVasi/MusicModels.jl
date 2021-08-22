using JLD2
using DataDeps
using Requires
using Transformers
using Pkg.TOML

export list_pretrains, load_pretrained, @pretrained_str

const pretrained_configs = open(TOML.parse, joinpath(@__DIR__, "pretrains.toml"))

"""
    readckpt(path)

Load weights from a tensorflow checkpoint file into a Dict.
"""
readckpt(path) = error("readckpt require TensorFlow.jl installed. run `Pkg.add(\"TensorFlow\"); using TensorFlow`")

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
    import .TensorFlow
    function readckpt(path)
        weights = Dict{String, Array}()
        TensorFlow.init()
        ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
        shapes = ckpt.get_variable_to_shape_map()

        for (name, shape) ∈ shapes
            # Ignore training related data - e.g. learning rates
            # Also ignore scalars and other variables that aren't stored correctly in the checkpoint (shape == Any[])
            (occursin("training", name) || shape == Any[]) && continue
            weight = ckpt.get_tensor(name)
            if length(shape) == 2
                weight = collect(weight')
            end
            weights[name] = weight
        end

        weights
    end
end

"""
    ckpt2_to_jld2(ckptpath::String, ckptname::String, savepath::String)

Loads the pre-trained model weights from a TensorFlow checkpoint and saves to JLD2
"""
function ckpt_to_jld2(ckptpath::String, ckptname::String; savepath::String="./")
    weights = readckpt(joinpath(ckptpath, ckptname))
    jld2name = normpath(joinpath(savepath, ckptname[1:end-5]*".jld2"))
    @info "Saving the model weights to $jld2name"
    JLD2.@save jld2name weights
end

function register_configs(configs)
    for (model_name, config) in pairs(configs)
        model_desc = Transformers.Pretrain.description(config["description"], config["host"], config["link"])
        checksum = config["checksum"]
        url = config["url"]
        dep = DataDep(model_name, model_desc, url, checksum;
                      fetch_method=Transformers.Datasets.download_gdrive)
        DataDeps.register(dep)
    end
end

"""
    list_pretrains()

List all the available pre-trained models.
"""
function list_pretrains()
    println.(keys(pretrained_configs))
    return
end

loading_method(model_name) = throw(error("Loading method for this model is not defined."))

"""
    load_pretrained(path)

Loads a pre-trained Music Transformer model.
"""
function load_pretrained(model_name::String)
    if model_name ∉ keys(pretrained_configs)
        error("""Invalid model.
               Please try list_pretrains() to check the available pre-trained models""")
    end

    model_config = pretrained_configs[model_name]
    loader = loading_method(Val(Symbol(model_config["model_type"])))

    model_path = @datadep_str("$model_name/$model_name.jld2")
    if !endswith(model_path, ".jld2")
        error("""Invalid file. A .jld2 file is required to load the model.
                 If this is a tensorflow checkpoint file, run ckpt_to_jld2 to convert it.""")
    end

    JLD2.@load model_path weights
    loader(weights, model_config)
end

macro pretrained_str(name)
    :(load_pretrained($(esc(name))))
end