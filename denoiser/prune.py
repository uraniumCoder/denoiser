import copy

def get_nchannels(model):
    """
    Get the number of channels of each layer in the model.
    """
    nchannels = {'encoder': [], 'decoder': [], 'lstm': []}
    nchannels['encoder'] = [[layer[0].in_channels, layer[2].in_channels] for layer in model.encoder]
    nchannels['decoder'] = [layer[2].in_channels for layer in model.decoder]
    nchannels['lstm'] = [model.lstm.lstm.input_size, model.lstm.lstm.hidden_size]

    return nchannels

def scale_down(n_channels, p):
    """
    Scale down the number of channels in the model by a factor of p.
    """
    tgt_nchannels = copy.deepcopy(n_channels)
    for i in range(len(tgt_nchannels['encoder'])):
        for k in range(2):
            if i==0 and k==0:
                continue # can't prune the input layer
            tgt_nchannels['encoder'][i][k] = int(tgt_nchannels['encoder'][i][k] * p)

    for i in range(len(tgt_nchannels['decoder'])):
        tgt_nchannels['decoder'][i] = int(tgt_nchannels['decoder'][i] * p)

    for i in range(2):
        tgt_nchannels['lstm'][i] = int(tgt_nchannels['lstm'][i] * p)
    
    return tgt_nchannels

def importance_selection(prev_convs, next_convs, n_channel):
    """
    Select the channels to keep based on importance scores.
    """
    importance = torch.zeros(next_convs[0].in_channels, device=next_convs[0].weight.device)
    for conv in next_convs:
        weight = conv.weight.detach()
        if isinstance(conv, torch.nn.Conv1d):
            importance += (weight**2).sum(dim=(0,2))
        elif isinstance(conv, torch.nn.ConvTranspose1d):
            importance += (weight**2).sum(dim=(1,2))
    keep_channels = torch.argsort(importance, descending=True)[:n_channel].detach()
    return keep_channels

def prune(model, tgt_nchannels):
    """
    Prune the model using importance scores. (L2 on input channels)
    Currently doesn't support pruning the LSTM layer.
    """
    with torch.no_grad():
        cur_nchannels = get_nchannels(model)
        for i in range(len(model.encoder)):
            # prune first conv inputs layer (the skip channels)
            if i != 0:  # cannot prune input layer nor the lstm skip connection
                cur, tgt = cur_nchannels['encoder'][i][0], tgt_nchannels['encoder'][i][0]       
                assert cur >= tgt, "Cannot prune to more channels than exist."

                next_convs = model.encoder[i][0], model.decoder[-i][0]
                prev_convs = model.encoder[i-1][2], model.decoder[-i-1][2]
                keep_channels = importance_selection(prev_convs, next_convs, tgt)

                # last conv in encoder is glu layer, so we need to keep the channels in pairs
                duplicated_channels = torch.cat([keep_channels, keep_channels+cur])
                prev_convs[0].weight.set_(prev_convs[0].weight.detach()[duplicated_channels])
                prev_convs[0].bias.set_(prev_convs[0].bias.detach()[duplicated_channels])
                prev_convs[0].out_channels = tgt*2
                # last conv in decoder is transpose conv, so remove from 2nd dim
                prev_convs[1].weight.set_(prev_convs[1].weight.detach()[:, keep_channels])
                prev_convs[1].bias.set_(prev_convs[1].bias.detach()[keep_channels])
                prev_convs[1].out_channels = tgt
                for conv in next_convs:
                    conv.weight.set_(conv.weight.detach()[:, keep_channels])
                    conv.in_channels = tgt

            # prune middle connection in encoder
            cur, tgt = cur_nchannels['encoder'][i][1], tgt_nchannels['encoder'][i][1]
            assert cur >= tgt, "Cannot prune to more channels than exist."
            next_conv = model.encoder[i][2]
            prev_conv = model.encoder[i][0]
            keep_channels = importance_selection([prev_conv], [next_conv], tgt)

            prev_conv.weight.set_(prev_conv.weight.detach()[keep_channels])
            prev_conv.bias.set_(prev_conv.bias.detach()[keep_channels])
            prev_conv.out_channels = tgt
            next_conv.weight.set_(next_conv.weight.detach()[:, keep_channels])
            next_conv.in_channels = tgt

            # prune middle connection in decoder
            cur, tgt = cur_nchannels['decoder'][i], tgt_nchannels['decoder'][i]
            assert cur >= tgt, "Cannot prune to more channels than exist."
            next_conv = model.decoder[i][2]
            prev_conv = model.decoder[i][0]
            keep_channels = importance_selection([prev_conv], [next_conv], tgt)

            # first conv in decoder is glu layer too
            duplicated_channels = torch.cat([keep_channels, keep_channels+cur])
            prev_conv.weight.set_(prev_conv.weight.detach()[duplicated_channels])
            prev_conv.bias.set_(prev_conv.bias.detach()[duplicated_channels])
            prev_conv.out_channels = tgt*2
            # last conv in decoder is transpose
            next_conv.weight.set_(next_conv.weight.detach()[keep_channels])
            next_conv.in_channels = tgt        

    print("Pruning complete.")