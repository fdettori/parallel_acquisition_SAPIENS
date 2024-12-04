import time
import numpy as np
from pylsl import StreamInlet, resolve_stream

def main(duration, eeg_queue):
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # Check if any streams are found
    if len(streams) == 0:
        raise RuntimeError("No EEG stream found")

    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    print("Start receiving data...")
    # Get the number of channels
    info = inlet.info()
    channel_count = info.channel_count()

    buffer_duration = 0.02  # 20ms
    start_time = time.time()
    while time.time() - start_time < duration:
        samples = []
        while time.time() - start_time < buffer_duration:
            sample, timestamp = inlet.pull_sample(timeout=buffer_duration)
            if sample is None:
                continue
            samples.append([timestamp] + sample)
        for sample in samples:
            eeg_queue.put(sample)
        time.sleep(buffer_duration)
