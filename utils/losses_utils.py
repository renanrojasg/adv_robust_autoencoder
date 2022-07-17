# Extract features
def extract_features( loss_style_layers, style, stylized, style_features,
                      stylized_style_features, model, model_arch):

    if model_arch== "alexnet":
        model.classifier.extract_layers( x= model.normalize( style),
                                         feat_dict= style_features,
                                         feat_level= loss_style_layers)
        model.classifier.extract_layers( x= model.normalize( stylized),
                                         feat_dict= stylized_style_features,
                                         feat_level= loss_style_layers)
    elif model_arch== "vgg19":
        model.extract_layers( x= style, feat_dict= style_features,
                              feat_level= loss_style_layers)
        model.extract_layers( x= stylized, feat_dict= stylized_style_features,
                              feat_level= loss_style_layers)
    else: raise ValueError( "Undefined model architecture. Check 'model_arch' argument.")
    return

