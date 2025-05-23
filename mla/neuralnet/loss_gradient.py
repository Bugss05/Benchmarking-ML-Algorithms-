from ..metrics import gradient_automatic_weighted_bce,gradient_focal_loss_weighted,gradient_categorical_crossentropy

automatic_weighted_binary_crossentropy = gradient_automatic_weighted_bce
focal_loss = gradient_focal_loss_weighted
binary_crossentropy= gradient_categorical_crossentropy

def get_gradient_loss(name):
    """Returns loss function by the name."""
    try:
        return globals()[name]
    except KeyError:
        raise ValueError("Invalid metric function.")
