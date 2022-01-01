def write_log(writer, log_dict, step):
    for key, value in log_dict.items():
        writer.add_scalar(key, value, step)
