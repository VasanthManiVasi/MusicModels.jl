# MusicModels

MusicModels.jl provides standard generative music models and music datasets. The models in this package are implemented in pure Julia, and their pre-trained versions are also hosted, making them ready to use at any point of time.

## Installation

MusicModels.jl uses [NoteSequences.jl](https://github.com/VasanthManiVasi/NoteSequences.jl) internally. Please install both of them to use the package.

```julia
] add https://github.com/VasanthManiVasi/NoteSequences.jl
] add https://github.com/VasanthManiVasi/MusicModels.jl
```

## Example

```julia
using FileIO
using MusicModels
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Generating music from scratch using the Music Transformer
music_transformer = pretrained"unconditional_model_16"
midi = generate(music_transformer, numsteps=1500)
save("transformer_generated.mid", midi)

# Generating music from scratch using the PerformanceRNN
perfrnn = pretrained"perfrnn_dynamics"
midi = generate(perfrnn, numsteps=3000)
save("perfrnn_generated.mid", midi)
```

See `examples/Melody Conditioning` for an example of using the Melody-Conditioned Music Transformer to generate an accompaniment given a melody.

## Available Models

| Model Name                           | Function                                       |
|:-------------------------------------|:-----------------------------------------------|
| PerformanceRNN                       | [`pretrained"perfrnn"`](#)                     |
| PerformanceRNN with dynamics         | [`pretrained"perfrnn_dynamics"`](#)            |
| Unconditional Music Transformer      | [`pretrained"unconditional_model_16"`](#)      |
| Melody-Conditioned Music Transformer | [`pretrained"melody_conditioned_model_16"`](#) |

## Available Datasets

| Dataset Name | Function                     | Type                                               |
|:------------:|:-----------------------------:|:---------------------------------------------------:|
|MAESTRO       | [`dataset"MAESTRO_Raw"`](#)  | Raw                                                |
|MAESTRO LM    | [`dataset"MAESTRO_LM"`](#)   | Processed for Piano Performance Language Modelling |
|Lakh MIDI     | [`dataset"LakhMIDI_Raw"`](#) | Raw                                                |

## Training

To train a new music transformer or to fine-tune one of the available models on your own collection of midi files, the midi files must first be converted to one-hot indices, which can then be fed to the model as inputs. Please refer to `examples/MAESTRO Language Modelling/maestro_datagen.jl` for an example script on converting midifiles to model inputs. The example provided performs data generation for training a piano performance language model on the MAESTRO dataset.
Finally, see `examples/MAESTRO Language Modelling/maestro_train.jl` for a training script to train a language model on the processed MAESTRO dataset we obtained from data generation. The examples can be modified for training on your own data.
