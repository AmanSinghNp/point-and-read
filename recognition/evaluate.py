import editdistance


def compute_cer(prediction, ground_truth):
    """
    Character Error Rate = edit_distance(pred, gt) / len(gt)
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return editdistance.eval(prediction, ground_truth) / len(ground_truth)


def compute_wer(prediction, ground_truth):
    """
    Word Error Rate = edit_distance(pred_words, gt_words) / len(gt_words)
    """
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, gt_words) / len(gt_words)


def greedy_decode(output, vocab):
    """
    Greedy (best-path) CTC decoding.

    Args:
        output: Tensor of shape (T, num_classes) — log-probabilities for one sample.
        vocab: Vocabulary instance.
    Returns:
        Decoded string.
    """
    # Take argmax at each timestep
    indices = output.argmax(dim=1).tolist()
    return vocab.decode(indices, remove_blanks=True, collapse_repeats=True)


def batch_greedy_decode(outputs, vocab):
    """
    Decode an entire batch.

    Args:
        outputs: Tensor of shape (T, batch, num_classes).
        vocab: Vocabulary instance.
    Returns:
        List of decoded strings.
    """
    batch_size = outputs.size(1)
    results = []
    for b in range(batch_size):
        results.append(greedy_decode(outputs[:, b, :], vocab))
    return results


def evaluate(model, dataloader, vocab, device="cpu"):
    """
    Run evaluation over a full DataLoader and return average CER and WER.
    """
    import torch
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels, input_lengths, target_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)  # (T, batch, num_classes)

            predictions = batch_greedy_decode(outputs, vocab)

            # Reconstruct ground-truth strings from flat label tensor
            offset = 0
            for i, tlen in enumerate(target_lengths):
                gt_indices = labels[offset : offset + tlen].tolist()
                gt_text = vocab.decode(gt_indices, remove_blanks=False, collapse_repeats=False)
                pred_text = predictions[i]

                total_cer += compute_cer(pred_text, gt_text)
                total_wer += compute_wer(pred_text, gt_text)
                num_samples += 1
                offset += tlen

    avg_cer = total_cer / max(num_samples, 1)
    avg_wer = total_wer / max(num_samples, 1)
    return avg_cer, avg_wer


if __name__ == "__main__":
    # Quick sanity check
    pred = "hell0 werld"
    gt = "hello world"
    print(f"CER('{pred}', '{gt}') = {compute_cer(pred, gt):.4f}")
    print(f"WER('{pred}', '{gt}') = {compute_wer(pred, gt):.4f}")
