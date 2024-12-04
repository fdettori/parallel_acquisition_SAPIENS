from pylsl import resolve_stream
streams = resolve_stream()
print(f"Found {len(streams)} streams")
for stream in streams:
    print(f"Stream name: {stream.name()}, Stream type: {stream.type()}")
