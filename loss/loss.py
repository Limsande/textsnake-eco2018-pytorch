import numpy as np
import torch
import torch.nn.functional as F


def loss_fn(prediction, maps):
    # TODO Online hard negative mining for tr_loss

    tr_pred = prediction[:, :2]
    tr_true = maps[0]
    tcl_pred = prediction[:, 2:4]
    tcl_true = maps[1]

    # tcl_loss only takes pixels inside text region into account.
    # We can use tr_true (Nx512x512, 1=text region, 0=no text region) to mask
    # the two channels of tcl_pred (Nx2x512x512) one by one. This lets us
    # construct a tensor tcl_pred_inside_tr with the same dimensions as tcl_pred,
    # but only with the values from tcl_pred, where tr_true is 1. Values outside
    # the text region are 0.
    tcl_pred_inside_tr = torch.stack([tcl_pred[:, 0] * tr_true, tcl_pred[:, 1] * tr_true], dim=1)

    # Geometry loss only takes pixels inside tcl into account. Use tcl_true as
    # mask, like tr_true above. But this time, there is only one channel for
    # radii, sine, and cosine, each. So no need to stack.
    mask = tcl_true > 0
    radii_pred_inside_tcl = torch.masked_select(input=prediction[:, 4], mask=mask)
    radii_true = torch.masked_select(input=maps[2], mask=mask)
    cos_pred_inside_tcl = torch.masked_select(input=prediction[:, 5], mask=mask)
    cos_true = torch.masked_select(input=maps[3], mask=mask)
    sin_pred_inside_tcl = torch.masked_select(input=prediction[:, 6], mask=mask)
    sin_true = torch.masked_select(input=maps[4], mask=mask)

    tr_loss = F.cross_entropy(tr_pred, tr_true.long())
    tcl_loss = F.cross_entropy(tcl_pred_inside_tr, tcl_true.long())
    radii_loss = F.smooth_l1_loss(radii_pred_inside_tcl, radii_true)
    sin_loss = F.smooth_l1_loss(sin_pred_inside_tcl, sin_true)
    cos_loss = F.smooth_l1_loss(cos_pred_inside_tcl, cos_true)

    radii_loss = radii_loss if not np.isnan(radii_loss.clone().cpu().detach().numpy()) else 0
    cos_loss = cos_loss if not np.isnan(cos_loss.clone().cpu().detach().numpy()) else 0
    sin_loss = sin_loss if not np.isnan(sin_loss.clone().cpu().detach().numpy()) else 0

    return tr_loss + tcl_loss + radii_loss + sin_loss + cos_loss


if __name__ == '__main__':
    # The loss function should only take pixel inside the text region
    # (for tcl_loss), or inside the tcl (for geometry loss) into account.
    # We test this by first calculating the loss on random predictions and
    # ground truth, but with true tr and tcl masks where everything is hot, i.e.
    # all values contribute to the loss.
    # Then, we enlarge everything with random values, except the true tr and tcl
    # masks, which we enlarge with zeros. To correct for the changed cross entropy
    # of tr and tcl we substract it from the loss. If the masking
    # works properly, the added values do not contribute to the loss, i.e. the
    # loss does not change.
    with torch.no_grad():
        # Make tr and tcl masks where everything is hot, i.e.
        # loss takes every predicted value into account.
        tr_true = torch.ones((1, 2, 2))
        tcl_true = torch.ones_like(tr_true)

        # Make up a random ground truth
        radii_true = torch.randn_like(tr_true)
        cos_true = torch.randn_like(tr_true)
        sin_true = torch.randn_like(tr_true)

        # Make up random predictions
        tr_pred = torch.randn((1, 2, 2, 2))
        tcl_pred = torch.randn_like(tr_pred)
        radii_pred = torch.randn((1, 1, 2, 2))
        cos_pred = torch.randn_like(radii_pred)
        sin_pred = torch.randn_like(radii_pred)

        # Calculate loss
        prediction = torch.cat([tr_pred, tcl_pred, radii_pred, cos_pred, sin_pred], dim=1)
        groundtruth = [tr_true, tcl_true, radii_true, cos_true, sin_true]
        loss = loss_fn(prediction, groundtruth)

        # Now enlarge tr and tcl masks, but only with zeros.
        tr_true_padded = torch.zeros((1, 4, 4))
        tr_true_padded[0, :2, :2] = tr_true
        tcl_true_padded = torch.zeros_like(tr_true_padded)
        tcl_true_padded[0, :2, :2] = tcl_true

        # Also enlarge the ground truth, but with random numbers
        radii_true_padded = torch.randn_like(tr_true_padded)
        radii_true_padded[0, :2, :2] = radii_true
        cos_true_padded = torch.randn_like(tr_true_padded)
        cos_true_padded[0, :2, :2] = cos_true
        sin_true_padded = torch.randn_like(tr_true_padded)
        sin_true_padded[0, :2, :2] = sin_true

        # Now also enlarge the predictions with random numbers,
        # except for tr_pred (enlarge with 0) to retain the same
        # cross entropy of tr_true and tr_pred.
        tr_pred_padded = torch.zeros((1, 2, 4, 4))
        tr_pred_padded[0, :, :2, :2] = tr_pred
        tcl_pred_padded = torch.randn_like(tr_pred_padded)
        tcl_pred_padded[0, :, :2, :2] = tcl_pred
        radii_pred_padded = torch.randn((1, 1, 4, 4))
        radii_pred_padded[0, 0, :2, :2] = radii_pred
        cos_pred_padded = torch.randn_like(radii_pred_padded)
        cos_pred_padded[0, 0, :2, :2] = cos_pred
        sin_pred_padded = torch.randn_like(radii_pred_padded)
        sin_pred_padded[0, 0, :2, :2] = sin_pred

        # Calculate loss on enlarged tensors.
        prediction = torch.cat([tr_pred_padded, tcl_pred_padded, radii_pred_padded, cos_pred_padded, sin_pred_padded], dim=1)
        groundtruth = [tr_true_padded, tcl_true_padded, radii_true_padded, cos_true_padded, sin_true_padded]
        masked_loss = loss_fn(prediction, groundtruth)

        # Eliminate the changed cross entropy of tr and tcl
        # Fake masking tcl_pred
        tcl_pred_padded = torch.zeros_like(tcl_pred_padded)
        tcl_pred_padded[0, :, :2, :2] = tcl_pred
        loss -= F.cross_entropy(tr_pred, tr_true.long()) + F.cross_entropy(tcl_pred, tcl_true.long())
        masked_loss -= F.cross_entropy(tr_pred_padded, tr_true_padded.long()) + F.cross_entropy(tcl_pred_padded, tcl_true_padded.long())
    assert torch.isclose(loss, masked_loss), f'Loss not equal: {loss} vs. {masked_loss}'
