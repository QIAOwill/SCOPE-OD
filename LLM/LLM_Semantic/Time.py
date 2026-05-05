import time


def get_interval(time_seconds):
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = int(time_seconds % 60)
    return hours, minutes, seconds


def get_normal_time(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def output_duration(start_time, end_time, print_time=False):
    elapsed_time_seconds = end_time - start_time
    hours, minutes, seconds = get_interval(elapsed_time_seconds)
    formatted_time = f"{hours}h {minutes}m {seconds}s"
    normal_start_time = get_normal_time(start_time)
    normal_end_time = get_normal_time(end_time)

    if print_time:
        print("\n" + "-" * 100)
        print(f"Start time: {normal_start_time}")
        print(f"End time:   {normal_end_time}")
        print(f"Duration:   {formatted_time}")
        print("-" * 100 + "\n")

    return normal_start_time, normal_end_time, formatted_time
