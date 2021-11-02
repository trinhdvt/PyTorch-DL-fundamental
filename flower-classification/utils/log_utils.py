import wandb

# write log to tensoreboard and wandb


def write_log(writer, log_dict, step):
    for key, value in log_dict.items():
        writer.add_scalar(key, value, step)
        wandb.log({key: value}, step=step)
