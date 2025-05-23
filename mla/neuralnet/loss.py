from ..metrics import mse, logloss, mae, hinge, binary_crossentropy,automatic_weighted_binary_crossentropy,focal_loss_weighted
categorical_crossentropy = logloss
focal_loss = focal_loss_weighted

def get_loss(name):
    """Returns loss function by the name."""
    try:
        return globals()[name]
    except KeyError:
        raise ValueError("Invalid metric function.")
