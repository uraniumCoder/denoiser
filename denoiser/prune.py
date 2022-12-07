def get_nchannels(model):
    """
    Get the number of channels of each layer in the model.
    """
    nchannels = {'encoder': [], 'decoder': [], 'lstm': []}
    nchannels['encoder'] = [(layer[0].in_channels, layer[2].in_channels) for layer in model.encoder]
    nchannels['decoder'] = [(layer[0].in_channels, layer[2].in_channels) for layer in model.decoder]
    nchannels['lstm'] = (model.lstm.input_size, model.lstm.hidden_size)

    return nchannels

