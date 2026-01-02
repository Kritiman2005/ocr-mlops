def should_promote(metrics):
    return metrics["eval_loss"] < 1.5
